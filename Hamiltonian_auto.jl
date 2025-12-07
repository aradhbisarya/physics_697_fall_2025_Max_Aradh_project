using ITensors, ITensorMPS
using Plots
using Printf
using Base.Threads
using ProgressMeter
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1) # Disable block-sparse multithreading

#sites, flavors, colors
N = 10
F = 1
C = 3

#mass and coupling strength
m0 = .5
g = -.2

#site size
a = 1

#charge conserved adjustments
w = 1/(2*a*g)
J = (a*g) / 2
m = m0 / g
L = 0

sites = siteinds("S=1/2", (N*F*C), conserve_qns=true)

#Helper functions

function addOp!(arr, op, site)
    push!(arr, op)
    push!(arr, site)
end

function addArr!(arr1, arr2)
    arr1[1] *= arr2[1]
    arr1 = append!(arr1, arr2[2:end])
end

function product(A, B)
    ret = []
    for a in A
        for b in B
            push!(ret, combine(a, b))
        end
    end
    return ret
end

function combine(a, b)
    coeff = a[1] * b[1]
    ops = Any[a[2:end]..., b[2:end]...]

    return tuple(coeff, ops...)
end



function l(n, f, c) #site index calculation
    return ((n-1)*F*C) + (C*(f -1)) + c - 1
end


function QijN(n, i, j)
    ret = []
    temp_coeff = 1/sqrt(2)
    for f=1: F
        temp = []
        push!(temp, temp_coeff)
        addOp!(temp, "S+", l(n, f, i) + 1)
        addArr!(temp, BigLambda(n, f, i, j))
        addOp!(temp, "S-", l(n, f, j) + 1)
        push!(ret, tuple(temp...))
    end

    return ret
end

function QiiN(n, i)
    temp = 1
    ret = []
    for f=1: F
        temp *= (1/(2*sqrt(2*i*(i -1))))
        for c=1 : i - 1
            push!(ret, (temp, "Sz", l(n, f, c) + 1))
            push!(ret, (-1 * temp, "Sz", l(n, f, i) + 1))
        end
    end
    return ret
end

function BigLambda(n, f, i, j)
    ret = []
    temp = 1
    for k=l(n, f, j) : l(n, f, i) - 1
        temp *= -2
        addOp!(ret, "Sz", k + 1)
    end
    pushfirst!(ret, temp)
    return ret
end 

function QiNQiM(n, o)
    ret = []
    for i=2 : C 
        for j=1: i - 1
            A = QijN(n, j, i)
            B = QijN(o, i, j)
            append!(ret, product(A, B))
            A = QijN(n, i, j)
            B = QijN(o, j, i)
            append!(ret, product(A, B))
        end
    end
    for i=2 : C
        A = QiiN(n, i)
        B = QiiN(o, i)
        append!(ret, product(A, B))
    end

    return ret
end

function plot_observables(psi, N, F, C) #AI generated plot
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

function plot_correlations(psi) #AI generated plot
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

function plot_entanglement(psi, N_total) #AI generated plot
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

function number_op(psi, w, n)
    num = AutoMPO()

    for c=1: C
        num += (w/2), "Sz", l(2*n, 1, c) + 1
        num += (w/2), "Sz", l((2*n)-1, 1, c) + 1
    end

    return inner(psi', MPO(num, siteinds(psi)), psi)
end

function phase_diagram(steps)
    M = zeros(steps, steps)
    mass = 1
    coupling = 1
    mass_vals = range(-mass, mass, length=steps)
    coupling_vals = range(-coupling, coupling, length=steps)

    p = Progress(steps * steps, dt=0.5, desc="Simulation Progress: ", barglyphs=BarGlyphs("[=> ]"))

    m_elem = construct_mass(N, F, C)
    c_elem = construct_hopping(N, F, C)

    Threads.@threads :dynamic for i=1 : steps
        mass_step = mass_vals[i]
        for j=1 : steps
            mNew = mass_step/coupling_step
            wNew = 1/(2*a*coupling_step)
            JNew = (a*coupling_step) / 2

            coupling_step = coupling_vals[j]
            if abs(coupling_step) < 1e-6 
                M[j, i] = NaN
                next!(p)
                continue
            end
            H = MPO((m_elem * mNew) + (c_elem * wNew) + construct_electric(N, JNew) + construct_flux(N, L), sites)
            EGap = calc_energy_gap(sites, H, false)
    

            if isnan(EGap[1])
                M[j, i] = NaN
            else
                M[j, i] = EGap[2] - EGap[1]
            end

            next!(p)
        end
    end
    gr()

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
    display(hm)
end

function phase_diagram_mn(steps)
    M = zeros(steps * 4, div(steps, 2))
    mass = 20
    mass_vals = range(-mass, mass, length=steps*4)
    n_vals = [i for i in 1:steps if iseven(i)]
    print(n_vals)

    p = Progress(steps * steps * 2, dt=0.5, desc="Simulation Progress: ", barglyphs=BarGlyphs("[=> ]"))

    Threads.@threads :dynamic for i=1 : length(n_vals)
        wNew = 1/(2*a*1)
        JNew = (a*1) / 2
        n_step = n_vals[i]
        sitesNew = siteinds("S=1/2", (n_step*F*C), conserve_qns=true)
        m_elem = construct_mass(n_step, F, C)
        c_elem = construct_hopping(n_step, F, C)
        e_elem = construct_electric(n_step, JNew)
        f_elem = construct_flux(n_step, L)
        
        for j=1 : length(mass_vals)
            mass_step = mass_vals[j]
            mNew = mass_step
            H = MPO((m_elem * mNew) + (c_elem * wNew) + e_elem + f_elem, sitesNew)
            EGap = calc_energy_gap(n_step, F, C, sitesNew, H, false)
    

            if isnan(EGap[1])
                M[j, i] = NaN
            else
                M[j, i] = EGap[2] - EGap[1]
            end

            next!(p)
        end
    end
    gr()

    threshold = 1000 # needs to be larger than expected largest energy gap ~2 * mass
    data_trimmed = copy(M)
    data_trimmed[data_trimmed .> threshold] .= NaN
    print(M)

    hm = heatmap(
        mass_vals,      # X-axis values
        n_vals,  # Y-axis values
        data_trimmed,
        title = "Phase Diagram (Ground State Energy Gap)",
        ylabel = "site",
        xlabel = "Mass",
        na_color = :green,
        c = :viridis 
    )
    display(hm)
end

function construct_mass(NNew, FNew, CNew)
    Mass = AutoMPO()
    for n=1: NNew 
        i = isodd(n) ? -1 : 1
        coeff = i
        for f=1: FNew
            for c=1: CNew
                Mass += coeff, "Sz", l(n, f, c) + 1
                Mass += coeff, "Id", l(n, f, c) + 1
            end
        end
    end
    return Mass
end

function construct_hopping(NNew, FNew, CNew)
    Hopping = AutoMPO()
    for n=1: NNew-1
        for f=1: FNew
            for c=1: CNew
                for i=1 : 2
                    s1 = l(n+1, f, c)
                    s2 = l(n, f, c)
                    if i==1
                        temp = []
                        coeff = 1
                        addOp!(temp, "S+", s1 + 1)
                        for k=s2 : s1 - 1
                            coeff *= -2
                            addOp!(temp, "Sz", k + 1)
                        end
                        addOp!(temp, "S-", s2 + 1)
                        pushfirst!(temp, coeff)
                        Hopping += tuple(temp...)
                    else
                        temp = []
                        coeff = 1
                        addOp!(temp, "S+", s2 + 1)
                        for k=s2 : s1 - 1
                            coeff *= -2
                            addOp!(temp, "Sz", k + 1)
                        end
                        addOp!(temp, "S-", s1 + 1)
                        pushfirst!(temp, coeff)
                        Hopping += tuple(temp...)
                    end
                end
            end
        end
    end
    return Hopping
end

function construct_electric(NNew, JNew)
    Electric = AutoMPO()
    for n=1 : NNew - 1
        coeff = (NNew) - n
        for i in QiNQiM(n, n)
            new_coeff = i[1] * coeff * JNew
            Electric += tuple(new_coeff, i[2:end]...)
        end
    end
    for n=1 : NNew - 2
        for o=n+1 : NNew - 1
            coeff = (NNew) - o
            for i in QiNQiM(n, o)
                new_coeff = i[1] * 2 * JNew * coeff
                Electric += tuple(new_coeff, i[2:end]...)
            end 
        end
    end
    return Electric

end

function construct_flux(NNew, LNew)
    Flux = AutoMPO()
    for n1=1: NNew
        for n2=1: NNew
            for i in QiNQiM(n1, n2)
                new_coeff = i[1] * LNew
                Flux += tuple(new_coeff, i[2:end]...)
            end
        end
    end
    return Flux
end

function construct_hamiltonian(sites, NNew, FNew, CNew, m0New, aNew, gNew, LNew)
    mNew = m0New/gNew
    wNew = 1/(2*aNew*gNew)
    JNew = (aNew*gNew) / 2

    #Mass
    Mass = construct_mass(NNew, FNew, CNew, mNew) * mNew

    #Hopping
    
    Hopping = construct_hopping(NNew, FNew, CNew) * wNew

    #Electric
    
    Electric = construct_electric(NNew, JNew)

    #Flux

    Flux = construct_flux(NNew, LNew)

    return MPO(Hopping + Mass + Electric + Flux, sites)
end

function calc_energy_gap(NNew, FNew, CNew, sites, H, show)
    nsweeps = 15 # number of sweeps
    maxdim = [10,20,20, 100, 200, 200, 400, 800, 1000, 1000, 2000, 2000, 2000, 2000, 2000] # gradually increase states kept
    cutoff = [1E-5, 1E-5, 1E-8, 1E-12] # desired truncation error
    noise = [1E-4, 1E-5, 1E-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #noise = [1E-6, 1E-7, 1E-8, 0.0, 0.0, 0.0]

    state_array = [isodd(div(x-1, F*C) + 1) ? "Up" : "Dn" for x in 1:(NNew*FNew*CNew)]

    psi = MPS(sites, state_array)

    if show
        energy, psi0 = dmrg(H, psi; nsweeps, maxdim, noise, cutoff)
    else
        energy, psi0 = dmrg(H, psi; nsweeps, maxdim, cutoff, noise, outputlevel = 0)
    end

    if show
        println("Generating ansatz for Excited State (Meson)...")
    end

    psi_init_1 = copy(psi0)
    successful_flip = false

    # Loop through every bond in the lattice to find a "flippable" pair
    for b in 1 : (NNew*FNew*CNew - 1)
        # Try direction 1
        op1 = op("S+", sites[b])
        op2 = op("S-", sites[b+1])
        
        psi_test = apply([op1, op2], psi_init_1; cutoff=1E-12)
        
        if norm(psi_test) > 0.1 # If norm is non-zero, flip is valid
            if show
                println("Found flippable ↓↑ site at bond $b. Creating Meson...")
            end
            psi_init_1 = psi_test
            successful_flip = true
            break
        end
        
        # Try direction 2
        op3 = op("S-", sites[b])
        op4 = op("S+", sites[b+1])
        
        psi_test = apply([op3, op4], psi_init_1; cutoff=1E-12)
        
        if norm(psi_test) > 0.1
            if show
                println("Found flippable ↑↓ site at bond $b. Creating Meson...")
            end
            psi_init_1 = psi_test
            successful_flip = true
            break
        end
    end

    if !successful_flip
        return [NaN, NaN]
    end

    # Now safe to run DMRG
    # weight can be smaller now that the state is physical
    if show
        energy1, psi1 = dmrg(H, [psi0], psi_init_1; nsweeps, maxdim, cutoff, noise, weight=100.0)
    else
        energy1, psi1 = dmrg(H, [psi0], psi_init_1; nsweeps, maxdim, cutoff, noise, weight=100.0, outputlevel=0)
    end
    

    if show
        print("Ground State Energy = ")
        println(energy)
        print("Excited State Energy = ")
        println(energy1)
        print("Energy Gap = ")
        println(energy1 - energy)
    end 
    return [energy, energy1]

end

let 
    phase_diagram_mn(4)
    # H = construct_hamiltonian(sites, N, F, C, m0, a, g, L)
    # calc_energy_gap(sites,H, true)

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