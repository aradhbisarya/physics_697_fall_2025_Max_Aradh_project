using Distributed
using Plots
using Printf
using JLD2

if nprocs() == 1
    addprocs(max(1, Sys.CPU_THREADS - 2))
    println("Added workers, total processes: $(nprocs())")
end

@everywhere begin

    using ITensors, ITensorMPS
    using LinearAlgebra
    using ProgressMeter

    BLAS.set_num_threads(1)
    ITensors.Strided.set_num_threads(1) # Disable block-sparse multithreading
    #ITensors.disable_threaded_blocksparse()

    struct ModelParams
        N::Int
        F::Int
        C::Int
        a::Float64
        g::Float64
        m0::Float64
        L::Float64
        theta::Float64
    end

    struct SymOp
        name::String
        site::Int
    end

    struct SymTerm
        coeff::Float64
        ops::Vector{SymOp}
    end

    #process functions

    const OPERATOR_CACHE = Dict{String, Any}()

    function init_worker_cache(p::ModelParams)
        println("Worker $(myid()): Building Operators....")

        sites = siteinds("S=1/2", p.N * p.F * p.C, conserve_qns=true)
        empty!(OPERATOR_CACHE)
        OPERATOR_CACHE["sites"] = sites

        m_op = construct_mass_op(sites, p)
        h_op  = construct_hopping_op(sites, p)
        c_op = construct_color_op(sites, p)
        e_op_quad, e_op_lin = construct_electric_spin_op(sites, p)

        OPERATOR_CACHE["Mass"] = m_op
        OPERATOR_CACHE["Hopping"] = h_op
        OPERATOR_CACHE["Color"] = c_op
        OPERATOR_CACHE["Electric_quad"] = e_op_quad
        OPERATOR_CACHE["Electric_lin"] = e_op_lin

        if p.L != 0.0
            OPERATOR_CACHE["Flux"] = construct_flux_op(sites, p)
        end

        println("Worker $(myid()) ready!")
    end

    function solve_element(mass_val, theta_val, p::ModelParams)
        sites = OPERATOR_CACHE["sites"]

        m_op = OPERATOR_CACHE["Mass"] 
        h_op = OPERATOR_CACHE["Hopping"]
        c_op = OPERATOR_CACHE["Color"]
        e_op_quad = OPERATOR_CACHE["Electric_quad"]
        e_op_lin = OPERATOR_CACHE["Electric_lin"]

        w = 1.0 / (2.0 * p.a * p.g)
        J = (p.a * p.g) / 2.0

        H = (h_op * w) +
            (c_op * J) +
            (e_op_quad * J) +
            (e_op_lin * (J * (theta_val / pi))) +
            (m_op * (mass_val/p.g))

        if p.L != 0.0
            H += OPERATOR_CACHE["Flux"] * p.L
        end

        gaps, psi0 = calc_energy_gap(p, sites, H, false)

        GC.gc() 
        return (gaps, psi0, sites)
    end

    function run_worker_task(N::Int, v::Float64, base::ModelParams)
        pN = ModelParams(
            N, base.F, base.C, v / N, 
            base.g, base.m0, base.L, base.theta
        )

        init_worker_cache(pN)

        gaps, psi = solve_element(pN.m0, pN.theta, pN)
        
        gap_val = NaN
        if isnan(gaps[1])
            gap_val = NaN
        else
            gap_val = gaps[2] - gaps[1]
        end

        cond_val = measure_chiral_condensate(psi, pN)

        flux_op_mpo = MPO(measure_total_flux_squared(pN), OPERATOR_CACHE["sites"])
        charge_val = inner(psi', flux_op_mpo, psi)
        baryon_val = number_op(pN, psi)

        psi = nothing
        GC.gc()

        # Return a lightweight tuple
        return (cond=cond_val, charge=charge_val, baryon=baryon_val, gap=gap_val)
    end

    #Helper functions

    l(n, f, c, p::ModelParams) = ((n-1) * p.F * p.C) + (p.C * (f - 1)) + c - 1

    # Multiply two symbolic terms (expands A * B)
    function multiply_terms(A::Vector{SymTerm}, B::Vector{SymTerm})
        ret = Vector{SymTerm}()
        sizehint!(ret, length(A) * length(B))
        for a in A
            for b in B
                new_coeff = a.coeff * b.coeff
                new_ops = vcat(a.ops, b.ops)
                push!(ret, SymTerm(new_coeff, new_ops))
            end
        end
        return ret
    end


    function QijN(n, i, j, p::ModelParams)
        ret = Vector{SymTerm}()
        pre_coeff = 1.0/sqrt(2)
        
        for f in 1:p.F
            lambda = BigLambda(n, f, i, j, p)
            
            # S+ at l(n,f,i), BigLambda, S- at l(n,f,j)
            new_ops = Vector{SymOp}()
            push!(new_ops, SymOp("S+", l(n, f, i, p) + 1))
            append!(new_ops, lambda.ops)
            push!(new_ops, SymOp("S-", l(n, f, j, p) + 1))
            
            final_coeff = pre_coeff * lambda.coeff
            push!(ret, SymTerm(final_coeff, new_ops))
        end
        return ret
    end

    function QiiN(n, i, p::ModelParams)
        ret = Vector{SymTerm}()
        val = 1.0
        
        for f in 1:p.F
            val *= (1.0 / (2.0 * sqrt(2.0 * i * (i - 1))))
            for c in 1:(i - 1)
                push!(ret, SymTerm(val, [SymOp("Sz", l(n, f, c, p) + 1)]))
                push!(ret, SymTerm(-val, [SymOp("Sz", l(n, f, i, p) + 1)]))
            end
        end
        return ret
    end

    function BigLambda(n, f, i, j, p::ModelParams)
        # Generates the string of Sz operators
        coeff = 1.0
        ops = Vector{SymOp}()
        
        start_idx = l(n, f, j, p)
        end_idx = l(n, f, i, p) - 1
        
        for k in start_idx:end_idx
            coeff *= -2.0
            push!(ops, SymOp("Sz", k + 1))
        end
        return SymTerm(coeff, ops)
    end 

    function QiNQiM(n, o, p::ModelParams)
        ret = Vector{SymTerm}()
        for i in 2:p.C
            for j in 1:(i - 1)
                append!(ret, multiply_terms(QijN(n, j, i, p), QijN(o, i, j, p)))
                append!(ret, multiply_terms(QijN(n, i, j, p), QijN(o, j, i, p)))
            end
        end
        for i in 2:p.C
            append!(ret, multiply_terms(QiiN(n, i, p), QiiN(o, i, p)))
        end
        return ret
    end

    function construct_mass_op(sites, p::ModelParams)
        os = OpSum()
        for n in 1:p.N
            sign_val = isodd(n) ? -1.0 : 1.0
            # Note: We do NOT multiply by mass here, we do it in the loop later
            coeff = sign_val 
            for f in 1:p.F
                for c in 1:p.C
                    idx = l(n, f, c, p) + 1
                    os += coeff, "Sz", idx
                    os += coeff, "Id", idx
                end
            end
        end
        return MPO(os, sites)
    end

    function construct_hopping_op(sites, p::ModelParams)
        os = OpSum()
        for n in 1:(p.N - 1)
            for f in 1:p.F
                for c in 1:p.C
                    s1 = l(n+1, f, c, p)
                    s2 = l(n, f, c, p)
                    
                    # Term 1: S+(s1) ... S-(s2)
                    c1 = 1.0
                    ops1 = Any[]
                    push!(ops1, "S+", s1 + 1)
                    for k in s2:(s1-1)
                        c1 *= -2.0
                        push!(ops1, "Sz", k + 1)
                    end
                    push!(ops1, "S-", s2 + 1)
                    # Combine coeff and ops into one tuple
                    os += (c1, ops1...)
                    
                    # Term 2: S+(s2) ... S-(s1)
                    c2 = 1.0
                    ops2 = Any[]
                    push!(ops2, "S+", s2 + 1)
                    for k in s2:(s1-1)
                        c2 *= -2.0
                        push!(ops2, "Sz", k + 1)
                    end
                    push!(ops2, "S-", s1 + 1)
                    os += (c2, ops2...)
                end
            end
        end
        return MPO(os, sites)
    end

    function construct_electric_spin_op(sites, p::ModelParams)
        os = OpSum()
        os_2 = OpSum()

        for n=1 : p.N - 1
            ops = Vector{SymTerm}()
            k_coeff = -1.0
            for k=1 : n
                coeff = 0.5
                for f=1 : p.F
                    for c=1 : p.C
                        idx = l(k, f, c, p) + 1
                        push!(ops, SymTerm(coeff, [SymOp("Sz", idx)]))
                    end 
                end
                coeff_2 = 0.5 * k_coeff * p.F * p.C
                push!(ops, SymTerm(coeff_2, [SymOp("Id", l(k, 1, 1, p) + 1)]))
                k_coeff *= -1.0
            end
            #construct linear term
            for t in ops
                final_coeff = t.coeff
                if abs(final_coeff) > 1e-14
                    args = Any[final_coeff]
                    for op in t.ops
                        push!(args, op.name, op.site)
                    end
                    os_2 += Tuple(args)
                end
            end
            #construct quadratic term
            for t in multiply_terms(ops, ops)
                final_coeff = t.coeff
                if abs(final_coeff) > 1e-14
                    args = Any[final_coeff]
                    for op in t.ops
                        push!(args, op.name, op.site)
                    end
                    os += Tuple(args)
                end
            end
        end
        return MPO(os, sites), MPO(os_2, sites)
    end

    function construct_color_op(sites, p::ModelParams)
        os = OpSum()

        # Internal helper to add terms to local os
        function add_terms_locally!(terms, factor)
            for t in terms
                final_coeff = t.coeff * factor
                if abs(final_coeff) > 1e-14
                    args = Any[final_coeff]
                    for op in t.ops
                        push!(args, op.name, op.site)
                    end
                    os += Tuple(args)
                end
            end
        end

        for n in 1:(p.N - 1)
            coeff = Float64(p.N - n)
            terms = QiNQiM(n, n, p)
            add_terms_locally!(terms, coeff)
        end
        for n in 1:(p.N - 2)
            for o in (n + 1):(p.N - 1)
                coeff = Float64(p.N - o)
                terms = QiNQiM(n, o, p)
                add_terms_locally!(terms, coeff * 2)
            end
        end
        return MPO(os, sites)
    end

    function construct_flux_op(sites, p::ModelParams)
        os = OpSum()

        function add_terms_locally!(terms, factor)
            for t in terms
                final_coeff = t.coeff * factor
                if abs(final_coeff) > 1e-14
                    args = Any[final_coeff]
                    for op in t.ops
                        push!(args, op.name, op.site)
                    end
                    os += Tuple(args)
                end
            end
        end

        for n1 in 1:p.N
            for n2 in 1:p.N
                terms = QiNQiM(n1, n2, p)
                add_terms_locally!(terms, 1)
            end
        end
        return MPO(os, sites)
    end

    function construct_hamiltonian(p, s)
        params = p
        sites = s

        m = params.m0/params.g
        w = 1.0 / (2 * params.a * params.g)
        J = (params.a * params.g) / 2.0

        #Mass
        Mass = construct_mass_op(sites, params) * m

        #Hopping
        
        Hopping = construct_hopping_op(sites, params) * w

        #Color
        
        Color = construct_color_op(sites, params) * J

        #Electric

        Electric_quad, Electric_lin = construct_electric_spin_op(sites, params)
        Electric = (Electric_quad * J) + (Electric_lin * (J * (p.theta / pi)))

        #Flux
        if p.L != 0.0
            Flux = construct_flux_op(sites, params) * p.L
            return Mass + Hopping + Electric + Color + Flux
        end

        return Mass + Hopping + Electric + Color
    end

    function measure_total_flux_squared(p)
        if p.L == 0.0
            factor = 1.0
        else
            factor = p.L
        end
        os = OpSum()
        for n1 in 1:p.N
            for n2 in 1:p.N
                terms = QiNQiM(n1, n2, p) 
                for t in terms
                    final_coeff = t.coeff * factor
                    if abs(final_coeff) > 1e-14
                        args = Any[final_coeff]
                        for op in t.ops
                            push!(args, op.name, op.site)
                        end
                        os += Tuple(args)
                    end
                end
            end
        end
        
        # println("Total Outgoing Flux Squared <Q^2>: $val")
        return os
    end

    function measure_chiral_condensate(psi, p::ModelParams)
        sz_exp = expect(psi, "Sz")
        
        total_val = 0.0
        for n in 1:p.N
            parity = isodd(n) ? -1.0 : 1.0
            
            for f in 1:p.F
                for c in 1:p.C
                    idx = l(n, f, c, p) + 1
                    total_val += parity * sz_exp[idx]
                end
            end
        end
        
        return abs(total_val / p.N) 
    end

    function calc_energy_gap(p::ModelParams, sites, H, show_output::Bool)
        nsweeps = 15
        maxdim = [10, 20, 50, 100, 200, 400, 800, 1000, 2000, 4000]
        cutoff = [1E-6, 1E-8, 1E-10, 1E-12]
        noise = [1E-4, 1E-5, 1E-6, 1E-8, 0.0]
    
        total_sites = length(sites)
        state_array = [isodd(div(x-1, p.F * p.C) + 1) ? "Up" : "Dn" for x in 1:total_sites]
        psi = MPS(sites, state_array)
    
        output_lvl = show_output ? 1 : 0
        energy0, psi0 = dmrg(H, psi; nsweeps, maxdim, cutoff, noise, outputlevel=output_lvl)
    
        if show_output; println("Generating ansatz for Excited State..."); end
    
        psi_exc = copy(psi0)
        successful_flip = false
    
        # Loop through every bond to find a "flippable" pair
        for b in 1:(total_sites - 1)
            # Try direction 1
            psi_test = apply([op("S+", sites[b]), op("S-", sites[b+1])], psi_exc; cutoff=1E-12)
            if norm(psi_test) > 0.1
                psi_exc = psi_test
                successful_flip = true
                break
            end
            # Try direction 2
            psi_test = apply([op("S-", sites[b]), op("S+", sites[b+1])], psi_exc; cutoff=1E-12)
            if norm(psi_test) > 0.1
                psi_exc = psi_test
                successful_flip = true
                break
            end
        end
    
        if !successful_flip
            return [NaN, NaN]
        end
    
        energy1, _ = dmrg(H, [psi0], psi_exc; nsweeps, maxdim, cutoff, noise, weight=100.0, outputlevel=output_lvl)
        return [energy0, energy1], psi0
    end

    function number_op(params::ModelParams, psi)
        baryon_number = 0.0
        w = 1.0 / (2.0 * params.a * params.g)
        for n=1: div(params.N, 2)
            num = AutoMPO()

            for c=1: params.C
                num += (w/2), "Sz", l(2*n, 1, c, params) + 1
                num += (w/2), "Sz", l((2*n)-1, 1, c, params) + 1
            end
            baryon_number += inner(psi', MPO(num, siteinds(psi)), psi)
        end
        return baryon_number
    end
    function calc_entanglement(p::ModelParams, psi)
        entropies = Float64[]
        # Orthogonalize to the first site to start
        orthogonalize!(psi, 1)
        
        for b in 1:(p.N-1)
            # Singular Value Decomposition at bond b
            # ITensor manages the orthogonality center automatically if you move sequentially
            orthogonalize!(psi, b)
            
            # Get the singular values (spectrum of the density matrix)
            _, S, _ = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
            
            # Calculate Von Neumann Entropy: - sum(p * log(p))
            Sv = 0.0
            for n in 1:dim(S, 1)
                p = S[n, n]^2
                if p > 1e-12 # Avoid log(0)
                    Sv -= p * log(p)
                end
            end
            push!(entropies, Sv)
        end

        return entropies
    end

    function calc_chiral_condensate(p::ModelParams, psi, sites)
        flux_op = MPO(measure_total_flux_squared(p), sites)

        val_charge = inner(psi', flux_op, psi)
        val_condensate = measure_chiral_condensate(psi, p)

        return (val_charge, val_condensate)
    end

end
function plot_observables(psi, p::ModelParams)
    sz = expect(psi, "Sz") 
    
    # Initialize plot
    plt = plot(title="Local Magnetization <Sz>", 
             xlabel="Physical Lattice Site (n)", 
             ylabel="<Sz>",
             legend=:outertopright)
    
    # Loop over Flavors and Colors
    # Assuming the mapping is: (n-1)*F*C + (f-1)*C + c
    markers = [:circle, :square, :diamond] # Different shapes for flavors
    
    for f in 1:p.F
        for c in 1:p.C
            # Extract indices for this specific flavor/color combination
            indices = [l(n, f, c, p) + 1 for n in 1:p.N]
            
            label_str = "F=$f, C=$c"
            
            plot!(plt, 1:p.N, sz[indices], 
                label=label_str, 
                marker=markers[mod1(f, 3)], 
                linewidth=1.5,
                alpha=0.8
            )
        end
    end
    
    # Add a zero line for reference
    hline!(plt, [0.0], color=:black, linestyle=:dash, label=nothing)
    
    display(plt)
end

function plot_correlations(psi) 
    M = correlation_matrix(psi, "Sz", "Sz")
    
    # Mask the diagonal to see off-diagonal structure
    for i in 1:size(M,1)
        M[i,i] = NaN
    end

    hm = heatmap(M, 
        title="Spin-Spin Correlation <Sz_i Sz_j>", 
        xlabel="Site j", 
        ylabel="Site i",
        color=:balance, # Diverging colormap
        clims=(-0.25, 0.25), # Fix scale to max possible range
        aspect_ratio=:equal,
        yflip=true # Matrix convention (index 1 at top)
    )
    display(hm)
end

# DEPRECIATED

# function number_op(psi, w, n, params)
#     num = AutoMPO()

#     for c=1: C
#         num += (w/2), "Sz", l(2*n, 1, c, params) + 1
#         num += (w/2), "Sz", l((2*n)-1, 1, c, params) + 1
#     end

#     return inner(psi', MPO(num, siteinds(psi)), psi)
# end

function plot_entanglement(p::ModelParams, filename)
    load_name = filename * ".JLD2"
    @load load_name psi
    site = 0

    tasks = collect(CartesianIndices((size(psi, 1), size(psi, 2))))
    
    results = @showprogress pmap(tasks) do idx
        i, j = idx.I
        return calc_entanglement(p, psi[i, j])
    end
    plotlyjs()

    all_entropies = Vector{Float64}[] 
    site_indices = Int[]

    lin_map = LinearIndices(psi)

    for (idx, res) in zip(tasks, results)
        site = lin_map[idx] % 6
        if site == 0
            site = 6
        end
        push!(all_entropies, res)
        push!(site_indices, site)
    end  

    z_matrix = reduce(hcat, all_entropies)
    x_axis = 1:size(z_matrix, 1) 
    y_axis = site_indices

    plt = surface(
        x_axis, 
        y_axis, 
        z_matrix, 
        title = "Entanglement Entropy (Von Neumann)",
        xlabel = "Bond Index",
        ylabel = "Site Index",
        zlabel = "Entropy S_vn",
        color = :viridis,   # Gradient colors are much easier to read in 3D
        colorbar = true,
        size = (800, 800) # Set a nice window size
    )

    savefig(plt, "entropy_interactive.html")
    display(plt)
end

function plot_chiral_condensate(p::ModelParams, filename)
    load_name = filename * ".JLD2"
    psi, sites, mass_vals, theta_vals = load(load_name, "psi", "sites", "mass_vals", "theta_vals")
    site = 0

    tasks = collect(CartesianIndices((size(psi, 1), size(psi, 2))))
    
    results = @showprogress pmap(tasks) do idx
        i, j = idx.I
        return calc_chiral_condensate(p, psi[i, j], sites[i, j])
    end

    M_condensate = zeros(size(psi, 1), size(psi, 2))
    M_charge = zeros(size(psi, 1), size(psi, 2))

    for (idx, res) in zip(tasks, results)
        i, j = idx.I

        M_charge[i, j] = res[1]
        M_condensate[i, j] = res[2]
    end  

    plot = heatmap(mass_vals, theta_vals, M_condensate, 
    title = "Chiral Condensate Phase Diagram",
    xlabel = "Mass Parameter (m)",
    ylabel = "Theta",
    color = :viridis, # Thermal is good for 0 to 1 intensity
    dpi = 300
    )
    filename = "chiral_condensate_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)

    jldsave(filename * ".jld2"; 
    M_condensate = M_condensate,
    M_charge = M_charge
    )
    savefig(filename * ".png")
    plot2 = heatmap(mass_vals, theta_vals, M_charge, 
    title = "Chiral Condensate Phase Diagram (charge)",
    xlabel = "Mass Parameter (m)",
    ylabel = "Theta",
    color = :viridis,
    dpi = 300
    )
    savefig(filename * "_charge" * ".png")
end

function plot_baryon_number(p::ModelParams, filename)
    load_name = filename * ".JLD2"
    psi, mass_vals, theta_vals = load(load_name, "psi", "mass_vals", "theta_vals")
    site = 0

    tasks = collect(CartesianIndices((size(psi, 1), size(psi, 2))))
    
    results = @showprogress pmap(tasks) do idx
        i, j = idx.I
        return number_op(p, psi[i, j])
    end

    BaryonNumber = zeros(size(psi, 1), size(psi, 2))

    for (idx, res) in zip(tasks, results)
        i, j = idx.I
        BaryonNumber[i, j] = res
    end  


    filename = "baryon_number_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)
    plot3 = heatmap(mass_vals, theta_vals, BaryonNumber, 
    title = "Baryon Number Phase Diagram",
    xlabel = "Mass Parameter (m)",
    ylabel = "Theta",
    na_color = :green,
    color = :viridis,
    dpi = 300
    )

    filename = "energy_gap_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)

    jldsave(filename * "_BaryonNumber.jld2"; 
    BaryonNumber = BaryonNumber
    )
    savefig(filename * ".png")
end

function phase_diagram_cached(steps, p)
    
    # A. INITIALIZE WORKERS
    # =====================
    println("--- SETUP PHASE ---")
    println("Initializing MPOs on all workers. Please wait 1-2 minutes...")
    
    # This triggers the 'init_worker_cache' function on every core.
    # It will block here until all workers are done building.
    @everywhere init_worker_cache($p) 
    
    println("--- SIMULATION PHASE ---")
    
    # B. RUN PARAMETER SWEEP
    # ======================
    mass_vals = collect(range(-10.0, 10.0, length=steps))
    theta_vals = collect(range(0, 2 * pi, length=steps))
    tasks = collect(CartesianIndices((steps, steps)))
    
    results = @showprogress pmap(tasks) do idx
        i, j = idx.I
        return solve_element(mass_vals[j], theta_vals[i], p)
    end
    
    # C. PLOTTING
    # ===========
    M = zeros(steps, steps)
    psi = Matrix{MPS}(undef, steps, steps)
    sites = Matrix{Vector{Index{Vector{Pair{QN, Int64}}}}}(undef, steps, steps)
    
    for (idx, res) in zip(tasks, results)
        i, j = idx.I
        gaps = res[1]

        if isnan(gaps[1])
            M[i, j] = NaN
        else
            M[i, j] = gaps[2] - gaps[1]
        end
        psi[i, j] = res[2]
        sites[i, j] = res[3]
    end

    filename = "energy_gap_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)

    threshold = 1000 # needs to be larger than expected largest energy gap ~2 * mass
    data_trimmed = copy(M)
    data_trimmed[data_trimmed .> threshold] .= NaN

    hm = heatmap(
        mass_vals,      # X-axis values
        theta_vals,  # Y-axis values
        data_trimmed,
        title = "Phase Diagram (Ground State Energy Gap)",
        ylabel = "Theta",
        xlabel = "Mass",
        na_color = :green,
        color = :viridis,
        dpi = 300
    )

    jldsave(filename * ".jld2"; 
    data = data_trimmed, 
    psi = psi,
    sites = sites,
    mass_vals = mass_vals,
    theta_vals = theta_vals
    )
    savefig(filename * ".png")

    return M
end

function sweep_over_N_and_plot(N_vals::Vector{Int}, v_vals::Vector{Float64}, base::ModelParams)
    param_combinations = vec(collect(Iterators.product(N_vals, v_vals)))
    
    println("Starting sweep over $(length(param_combinations)) configurations...")

    # resulting_data will be a Vector of NamedTuples
    resulting_data = @showprogress pmap(param_combinations) do (N, v)
        return run_worker_task(N, v, base)
    end

    rows = length(N_vals)
    cols = length(v_vals)
    
    M_cond   = zeros(rows, cols)
    M_charge = zeros(rows, cols)
    M_baryon = zeros(rows, cols)
    M_gap    = zeros(rows, cols)

    for (idx, (N, v)) in enumerate(param_combinations)
        i = findfirst(==(N), N_vals)
        j = findfirst(==(v), v_vals)
        
        res = resulting_data[idx]
        
        M_cond[i, j]   = res.cond
        M_charge[i, j] = res.charge
        M_baryon[i, j] = res.baryon
        M_gap[i, j]    = res.gap
        
        @printf("N=%d, v=%.2f | <ψ̄ψ>=%.4f | <Q²>=%.4f | B=%.4f | ΔE=%.4f\n", 
                N, v, res.cond, res.charge, res.baryon, res.gap)
    end

    # --- PLOTTING ---
    # Initialize plots
    p1 = plot(xlabel = "N", ylabel = "⟨ψ̄ψ⟩", title = "Chiral Condensate vs N", legend=:topright)
    p2 = plot(xlabel = "N", ylabel = "⟨Q²⟩", title = "Charge vs N", legend=:topright)
    p3 = plot(xlabel = "N", ylabel = "Baryon Number", title = "Baryon vs N", legend=:topright)
    p4 = plot(xlabel = "N", ylabel = "ΔE", title = "Energy Gap vs N", legend=:topright)

    # Loop over v_vals (columns) to add a line for each v
    for j in 1:cols
        v = v_vals[j]
        lbl = "v = $(v)"
        
        plot!(p1, N_vals, M_cond[:, j],   marker=:circle, label=lbl)
        plot!(p2, N_vals, M_charge[:, j], marker=:square, label=lbl)
        plot!(p3, N_vals, M_baryon[:, j], marker=:diamond, label=lbl)
        plot!(p4, N_vals, M_gap[:, j],    marker=:utriangle, label=lbl)
    end

    # Combine and Save
    final_plot = plot(p1, p2, p3, p4, layout = (2, 2), size=(1000, 800))
    
    filename = "sweep_N_results"
    savefig(final_plot, filename * ".png")
    
    # Save raw data
    jldsave(filename * ".jld2"; 
        N_vals=N_vals, 
        v_vals=v_vals, 
        condensate=M_cond, 
        charge=M_charge, 
        baryon=M_baryon, 
        gap=M_gap
    )
    
    println("Sweep complete. Saved to $filename.png")
    display(final_plot)
end



let 
    params = ModelParams(10, 1, 2, 1.0, 1.0, 20.0, 0, 1)
    filename = "energy_gap_PD_N" *  string(params.N) * "_C" * string(params.C) * "_F" * string(params.F)
    # # phase_diagram_mn(16)
    #  phase_diagram_cached(20, params)
    plot_entanglement(params, filename)
    plot_chiral_condensate(params, filename)
    plot_baryon_number(params, filename)
    #phase_diagram_condensate(20, params)
    # sites = siteinds("S=1/2", params.N * params.F * params.C, conserve_qns=true)
    # H = construct_hamiltonian(params, sites)
    # calc_energy_gap(params, sites, H, true)

    # N_vals = [4, 12, 10]
    # v_vals = [0.1,1,10]

    # sweep_over_N_and_plot(N_vals, v_vals,params)


end