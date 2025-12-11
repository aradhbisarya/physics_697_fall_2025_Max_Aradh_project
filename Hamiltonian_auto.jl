using ITensors, ITensorMPS
using Plots
using Printf
using Base.Threads
using ProgressMeter
using LinearAlgebra
using JLD2

BLAS.set_num_threads(2)
ITensors.Strided.set_num_threads(2) # Disable block-sparse multithreading

struct ModelParams
    N::Int
    F::Int
    C::Int
    a::Float64
    g::Float64
    m0::Float64
    L::Float64
end

struct SymOp
    name::String
    site::Int
end

struct SymTerm
    coeff::Float64
    ops::Vector{SymOp}
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

function l(n, f, c) #site index calculation
    return ((n-1)*F*C) + (C*(f -1)) + c - 1
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

function construct_electric_op(sites, p::ModelParams)
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

    #Electric
    
    Electric = construct_electric_op(sites, params) * J

    #Flux
    if p.L != 0.0
        Flux = construct_flux_op(sites, params) * L
        return Mass + Hopping + Electric + Flux
    end

    return Mass + Hopping + Electric
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

function plot_observables(psi, N, F, C)
    # Calculate expectation value of Sz at every site
    sz = expect(psi, "Sz") 
    
    # Initialize a plot
    p = plot(title="Local Magnetization <Sz>", xlabel="Lattice Site (n)", ylabel="<Sz>")
    
    # Loop through "Colors" (C) and plot them as separate series
    for c_idx = 1:C
        # Extract only the indices corresponding to this color
        # We reconstruct the linear index: l(n, c) + 1
        indices = [((n-1)*F*C) + c_idx - 1 + 1 for n in 1:N]
        
        # Plot this subset
        plot!(p, 1:N, sz[indices], label="Color $c_idx", marker=:circle)
    end
    display(p)
end

function plot_correlations(psi) 
    # specific operators
    M = correlation_matrix(psi, "Sz", "Sz")
    
    # Plot as a heatmap
    hm = heatmap(M, 
        title="Spin-Spin Correlation <Sz_i Sz_j>", 
        xlabel="Site j", 
        ylabel="Site i",
        color=:viridis,
        aspect_ratio=:equal
    )
    display(hm)
end

function plot_entanglement(psi, N_total) 
    entropies = Float64[]
    
    # Orthogonalize to the first site to start
    orthogonalize!(psi, 1)
    
    for b in 1:(N_total-1)
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

    p = plot(entropies, 
        title="Entanglement Entropy (Von Neumann)", 
        xlabel="Bond Cut", 
        ylabel="Entropy S_vn", 
        legend=false,
        fill=(0, 0.2, :blue) # Fill area under curve
    )
    display(p)
end

function number_op(psi, w, n, params)
    num = AutoMPO()

    for c=1: C
        num += (w/2), "Sz", l(2*n, 1, c, params) + 1
        num += (w/2), "Sz", l((2*n)-1, 1, c, params) + 1
    end

    return inner(psi', MPO(num, siteinds(psi)), psi)
end

function phase_diagram(steps, p)
    M = zeros(steps, steps)
    mass = 5.0
    coupling = 1.5
    mass_vals = collect(range(-mass, mass, length=steps))
    coupling_vals = range(0.1, coupling, length=steps)

    p_meter = Progress(steps * steps, dt=0.5, desc="Simulation Progress: ", barglyphs=BarGlyphs("[=> ]"))

    sites = siteinds("S=1/2", p.N * p.F * p.C, conserve_qns=true)

    m_elem = construct_mass_op(sites, p)
    c_elem = construct_hopping_op(sites, p)
    e_elem = construct_electric_op(sites, p)

    # os = measure_total_flux_squared(p)

    for i=1 : steps
        mass_step = mass_vals[i]

        Threads.@threads :dynamic for j=1 : steps
            coupling_step = coupling_vals[j]
            mNew = mass_step/coupling_step
            wNew = 1/(2*p.a*coupling_step)
            JNew = (p.a*coupling_step) / 2


            if abs(coupling_step) < 1e-6 
                M[j, i] = NaN
                next!(p)
                continue
            end
            H = (m_elem * mNew) + (c_elem * wNew) + (e_elem * JNew)
            EGap = calc_energy_gap(p, sites, H, false)
    

            if isnan(EGap[1])
                M[j, i] = NaN
            else
                M[j, i] = EGap[2] - EGap[1]
            end

            next!(p_meter)
        end
    end

    threshold = 1000 # needs to be larger than expected largest energy gap ~2 * mass
    data_trimmed = copy(M)
    data_trimmed[data_trimmed .> threshold] .= NaN
    print(M)

    hm = heatmap(
        mass_vals,      # X-axis values
        coupling_vals,  # Y-axis values
        data_trimmed,
        title = "Phase Diagram (Ground State Energy Gap)",
        ylabel = "Coupling Strength",
        xlabel = "Mass",
        na_color = :green,
        c = :viridis 
    )

    filename = "Energy_Gap_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)

    jldsave(filename * ".jld2"; 
    gaps = data_trimmed
    )
    savefig(filename * ".png")
    display(hm)
    return data_trimmed
end

function phase_diagram_mn(steps)
    # Define Base Parameters
    F, C = 1, 3
    a_val, g_val, m0_val, L_val = 1.0, 1.0, 1.0, 0.0
    
    mass_limit = 8.0
    mass_vals = collect(range(-mass_limit, mass_limit, length=steps*4))
    n_vals = [i for i in 1:steps if iseven(i)]
    
    # Pre-allocate matrix (rows=mass, cols=sites)
    results = zeros(Float64, length(mass_vals), length(n_vals))

    p_meter = Progress(length(n_vals) * length(mass_vals), desc="Simulation Progress: ", barglyphs=BarGlyphs("[=> ]"))

    for i in 1:length(n_vals)
        local_n = n_vals[i]

        params = ModelParams(local_n, F, C, a_val, g_val, m0_val, L_val)

        # Constants
        w = 1.0 / (2 * params.a * params.g)
        J = (params.a * params.g) / 2.0
        
        local_sites = siteinds("S=1/2", local_n * params.F * params.C, conserve_qns=true)

        m_op = construct_mass_op(local_sites, params)
        h_op  = construct_hopping_op(local_sites, params)
        e_op = construct_electric_op(local_sites, params)
        #f_op = construct_flux_op(local_sites, params)


        H_fixed = (w * h_op) + e_op * J
        
        Threads.@threads :dynamic for j in 1:length(mass_vals)
            mass_step = mass_vals[j]

            H = (mass_step * m_op) + H_fixed
            gaps = calc_energy_gap(params, local_sites, H, false)
    

            if isnan(gaps[1])
                results[j, i] = NaN
            else
                results[j, i] = gaps[2] - gaps[1]
            end

            next!(p_meter)
        end
    end

    # ... loops finished ...

    threshold = 1000.0
    results_to_save = copy(results)
    results[results .> threshold] .= NaN

    println("Saving simulation data to phase_diagram.jld2...")
    jldsave("phase_diagram.jld2"; 
        results=results_to_save, 
        mass_vals=mass_vals, 
        n_vals=n_vals
    )

    # ... proceed to plotting ...

    hm = heatmap(
        mass_vals,      # X-axis values
        n_vals,  # Y-axis values
        results_to_save',
        title = "Phase Diagram (Ground State Energy Gap)",
        ylabel = "Sites (N)",
        xlabel = "Mass",
        na_color = :green,
        c = :viridis 
    )


    display(hm)
    return results
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

function phase_diagram_condensate(steps, p)
    
    M_condensate = zeros(steps, steps)
    M_charge = zeros(steps, steps)
    
    mass_range = range(-5.0, 5.0, length=steps)
    g_range = range(0.1, 1.5, length=steps) # Avoid g=0 to prevent division by zero
    sites = siteinds("S=1/2", p.N * p.F * p.C, conserve_qns=true)
    
    println("Starting Phase Diagram Scan...")

    m_op = construct_mass_op(sites, p)
    h_op  = construct_hopping_op(sites, p)
    e_op = construct_electric_op(sites, p)
    #f_op = construct_flux_op(sites, params)

    p_meter = Progress(steps * steps, desc="Simulation Progress: ", barglyphs=BarGlyphs("[=> ]"))

    os = measure_total_flux_squared(p) 

    #DMRG Parameters
    nsweeps = 15
    maxdim = [10, 20, 50, 100, 200, 400, 1000, 2000]
    noise = [1E-4, 1E-5, 0.0]
    cutoff = [1E-6, 1E-8, 1E-12]

    Threads.@threads :dynamic for i in 1:steps
        w = 1.0 / (2 * p.a * g_range[i])
        J = (p.a * g_range[i]) / 2.0
       
        H_fixed = (h_op * w) + (e_op * J)
        for j in 1:steps
            m = mass_range[j] / g_range[i]
            H = (m_op * m) + H_fixed
            
            state_array = [isodd(div(x-1, p.F*p.C) + 1) ? "Up" : "Dn" for x in 1:(p.N*p.F*p.C)]
            psi_init = random_mps(sites, state_array)

            energy, psi0 = dmrg(H, psi_init; nsweeps, maxdim, cutoff, noise, outputlevel=0)

            flux_squared_op = MPO(os, siteinds(psi0))
    
            M_charge[j, i] = inner(psi0', flux_squared_op, psi0)

            #Measure Condensate
            val = measure_chiral_condensate(psi0, p)
            
            M_condensate[j, i] = val
            next!(p_meter)
        end
    end

    # Plotting
    plot = heatmap(mass_range, g_range, M_condensate', 
        title = "Chiral Condensate Phase Diagram",
        xlabel = "Mass Parameter (m)",
        ylabel = "Coupling Constant (g)",
        color = :viridis,
        aspect_ratio = :auto # Allows the plot to stretch to fill the window
    )
    filename = "chiral_condensate_PD_N" *  string(p.N) * "_C" * string(p.C) * "_F" * string(p.F)

    jldsave(filename * ".jld2"; 
    M_condensate = M_condensate, 
    M_charge = M_charge
    )
    savefig(filename * ".png")
    plot2 = heatmap(mass_range, g_range, M_charge', 
    title = "Chiral Condensate Phase Diagram (charge)",
    xlabel = "Mass Parameter (m)",
    ylabel = "Coupling Constant (g)",
    color = :viridis,
    aspect_ratio = :auto # Allows the plot to stretch to fill the window
    )
    savefig(filename * "_charge" * ".png")
    display(plot)
    return M_condensate
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
    return [energy0, energy1]
end

let 

    params = ModelParams(6, 1, 2, 1.0, 1.0, 20.0, 0)
    # phase_diagram_mn(16)
    phase_diagram(20, params)
    #phase_diagram_condensate(20, params)
    # sites = siteinds("S=1/2", params.N * params.F * params.C, conserve_qns=true)
    # H = construct_hamiltonian(params, sites)
    # calc_energy_gap(params, sites, H, true)

    # ret = []
    # for n=1: div(N,2)
    #     push!(ret, number_op(psi0, w, n))
    # end

    # ret2 = []
    # for n=1: div(N,2)
    #     push!(ret2, number_op(psi1, w, n))
    # end

    # p = plot(title="Number operator vaules for Random and Minimized states", xlabel="Site", ylabel="Number operator value")
    # plot!(p, ret, label="zero")
    # plot!(p, ret2, label="one")
    # display(p)
    
    # plot_observables(psi, N, F, C)

    # plot_correlations(psi)

    # plot_entanglement(psi, N*F*C)

end