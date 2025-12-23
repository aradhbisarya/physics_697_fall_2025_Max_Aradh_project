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

        OPERATOR_CACHE["Flux_measure"] = MPO(measure_total_flux_squared(p), sites)

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

        flux_op = OPERATOR_CACHE["Flux_measure"]
        val_charge = inner(psi0', flux_op, psi0)
        val_condensate = measure_chiral_condensate(psi0, p)
        baryon_number = number_op(psi0, p)
        GC.gc() 
        return (val_condensate, val_charge, gaps,baryon_number)
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

    function measure_chiral_condensate(psi, p)
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

    function number_op(psi, params)
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
    mass_vals = collect(range(-20.0, 20.0, length=steps))
    theta_vals = collect(range(1, 100.0, length=steps))
    tasks = collect(CartesianIndices((steps, steps)))
    
    results = @showprogress pmap(tasks) do idx
        i, j = idx.I
        # Workers are already prepped. We just send the coordinates.
        return solve_element(mass_vals[j], theta_vals[i], p)
    end
    
    # C. PLOTTING
    # ===========
    M = zeros(steps, steps)
    M_condensate = zeros(steps, steps)
    M_charge = zeros(steps, steps)
    BaryonNumber = zeros(steps, steps)
    for (idx, res) in zip(tasks, results)
        i, j = idx.I
        M_condensate[j, i] = res[1]
        M_charge[j, i] = res[2]
        gaps = res[3]
        BaryonNumber[j, i] = res[4]

        if isnan(gaps[1])
            M[j, i] = NaN
        else
            M[j, i] = gaps[2] - gaps[1]
        end
    end

    # ... [Copy your plotting code here] ...
    threshold = 1000 # needs to be larger than expected largest energy gap ~2 * mass
    data_trimmed = copy(M)
    data_trimmed[data_trimmed .> threshold] .= NaN

    hm = heatmap(
        mass_vals,      # X-axis values
        theta_vals,  # Y-axis values
        data_trimmed',
        title = "Phase Diagram (Ground State Energy Gap)",
        ylabel = "Theta",
        xlabel = "Mass",
        na_color = :green,
        color = :viridis,
        dpi = 300
    )

    filename = "energy_gap_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)

    jldsave(filename * ".jld2"; 
    data = data_trimmed
    )
    savefig(filename * ".png")

    plot = heatmap(mass_vals, theta_vals, M_condensate', 
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
    plot2 = heatmap(mass_vals, theta_vals, M_charge', 
    title = "Chiral Condensate Phase Diagram (charge)",
    xlabel = "Mass Parameter (m)",
    ylabel = "Theta",
    color = :viridis,
    dpi = 300
    )
    savefig(filename * "_charge" * ".png")

    filename = "baryon_number_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)
    plot3 = heatmap(mass_vals, theta_vals, BaryonNumber', 
    title = "Baryon Number Phase Diagram",
    xlabel = "Mass Parameter (m)",
    ylabel = "Theta",
    xlabel = "Mass",
    na_color = :green,
    color = :viridis,
    dpi = 300
    )

    filename = "energy_gap_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)

    jldsave(filename * ".jld2"; 
    BaryonNumber = BaryonNumber,
    data = data_trimmed
    )
    savefig(filename * ".png")
    return M
end
function sweep_over_N_and_plot(
    N_vals::Vector{Int},
    v_vals::Vector{Float64},
        base::ModelParams
    )# ---------- INITIALIZE PLOTS ONCE ----------
    p1 = plot(
        xlabel = "N",
        ylabel = "⟨ψ̄ψ⟩",
        title = "Chiral Condensate vs N"
    )

    p2 = plot(
        xlabel = "N",
        ylabel = "⟨Q²⟩",
        title = "Charge vs N"
    )

    p3 = plot(
        xlabel = "N",
        ylabel = "Baryon Number",
        title = "Baryon Number vs N"
    )

    p4 = plot(
        xlabel = "N",
        ylabel = "ΔE",
        title = "Energy Gap vs N"
    )

    results = Dict()

    for v in v_vals

        condensate = Float64[]
        charge     = Float64[]
        baryon     = Float64[]
        gap        = Float64[]

        for N in N_vals
            println("\n==============================")
            println("Running N = $N   (a = 1/$N, v = $v)")
            println("==============================")

            pN = ModelParams(
                N,
                base.F,
                base.C,
                v / N,
                base.g,
                base.m0,
                base.L,
                base.theta
            )

            @everywhere init_worker_cache($pN)

            val_cond, val_charge, gaps, baryon_val =
                solve_element(pN.m0, pN.theta, pN)

            gap_val = isnan(gaps[1]) ? NaN : gaps[2] - gaps[1]

            push!(condensate, val_cond)
            push!(charge,     val_charge)
            push!(baryon,     baryon_val)
            push!(gap,        gap_val)
            display(
                @sprintf(
                    "N=%d | ⟨ψ̄ψ⟩=%.6f | ⟨Q²⟩=%.6f | Baryon=%.6f | ΔE=%.6f",
                    N,
                    val_cond,
                    val_charge,
                    baryon_val,
                    gap_val
                )
            )
        end

        # ---------- OVERLAY CURVES ----------
        plot!(p1, N_vals, condensate, marker = :o, label = "v = $v")
        plot!(p2, N_vals, charge,     marker = :o, label = "v = $v")
        plot!(p3, N_vals, baryon,     marker = :o, label = "v = $v")
        plot!(p4, N_vals, gap,        marker = :o, label = "v = $v")

        results[v] = (
            N = N_vals,
            condensate = condensate,
            charge = charge,
            baryon = baryon,
            gap = gap
        )
    end

    # ---------- SAVE FIGURES ----------
    savefig(p1, "condensate_vs_N.png")
    savefig(p2, "charge_vs_N.png")
    savefig(p3, "baryon_vs_N.png")
    savefig(p4, "gap_vs_N.png")

    savefig(
        plot(p1, p2, p3, p4, layout = (2,2)),
        "combined_N_sweep.png"
    )

    return results
end



let 
    params = ModelParams(4, 1, 3, 1.0, 1.0, 1.0, 0, 1)
    # # phase_diagram_mn(16)
    # phase_diagram_cached(2, params)

    N_vals = [4, 12, 10]
    v_vals = [0.1,1,10]

    sweep_over_N_and_plot(N_vals, v_vals,params)


end