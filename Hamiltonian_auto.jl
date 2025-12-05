using ITensors, ITensorMPS
using Plots
using Printf
using Base.Threads
using ProgressMeter

#sites, flavors, colors
N = 20
F = 1
C = 3

#mass and coupling strength
m0 = 20
g = 1

#site size
a = 1

#charge conserved adjustments
w = 1/(2*a*g)
J = (a*g) / 2
m = m0 / g
L = 10000

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


function QijN(n, i, j, coeff)
    ret = []
    for f=1: F
        coeff = 1/sqrt(2)
        temp = []
        push!(temp, coeff)
        addOp!(temp, "S+", l(n, f, i) + 1)
        addArr!(temp, BigLambda(n, f, i, j))
        addOp!(temp, "S-", l(n, f, j) + 1)
        push!(ret, tuple(temp...))
    end

    return ret
end

function QiiN(n, i, coeff)
    coeff = 1
    ret = []
    for f=1: F
        coeff *= (1/(2*sqrt(2i*(i -1))))
        for c=1 : i - 1
            push!(ret, (coeff, "Sz", l(n, f, c) + 1))
            push!(ret, (-1 * coeff, "Sz", l(n, f, i) + 1))
        end
    end
    return ret
end

function BigLambda(n, f, i, j)
    ret = []
    coeff = 1
    for k=l(n, f, j) : l(n, f, i) - 1
        coeff *= -2
        addOp!(ret, "Sz", k + 1)
    end
    pushfirst!(ret, coeff)
    return ret
end 

function QiNQiM(n, m, coeff)
    ret = []
    for i=2 : C 
        for j=1: i - 1
            A = QijN(n, j, i, coeff)
            B = QijN(m, i, j, coeff)
            append!(ret, product(A, B))
            A = QijN(n, i, j, coeff)
            B = QijN(m, j, i, coeff)
            append!(ret, product(A, B))
        end
    end
    for i=2 : C
        A = QiiN(n, i, coeff)
        B = QiiN(m, i, coeff)
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

    Threads.@threads :dynamic for i=1 : steps
        mass_step = mass_vals[i]
        for j=1 : steps
            coupling_step = coupling_vals[j]
            if coupling_step == 0
                continue
            end
            # @printf("step %10d", (steps * (i - 1)) + j)
            # print("/")
            # @printf("%10d\n", steps * steps)
            # println(mass)
            # println(coupling)
            H = construct_hamiltonian(N, F, C, mass_step, a, coupling_step)
            EGap = calc_energy_gap(H, false)
    

            M[j, i] = EGap[2] - EGap[1]

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

function construct_hamiltonian(NNew, FNew, CNew, m0New, aNew, gNew)
    N = NNew
    F = FNew   
    C = CNew
    m0 = m0New
    a = aNew
    g = gNew
    m = m0/g
    w = 1/(2*a*g)
    J = (a*g) / 2

    #Mass
    Mass = AutoMPO()
    for n=1: N 
        i = isodd(n) ? -1 : 1
        coeff = m*i
        for f=1: F
            for c=1: C
                Mass += coeff, "Sz", l(n, f, c) + 1
                Mass += coeff, "Id", l(n, f, c) + 1
            end
        end
    end

    #Hopping
    Hopping = AutoMPO()
    for n=1: N-1
        for f=1: F
            for c=1: C
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

    #Electric
    Electric = AutoMPO()
    for n=1 : N - 1
        coeff = (N) - n
        for i in QiNQiM(n, n, J * coeff)
            Electric += i
        end
    end
    for n=1 : N - 2
        for m=n+1 : N - 1
            coeff = (N) - m 
            for i in QiNQiM(n, m, 2 * J * coeff)
                Electric += i
            end 
        end
    end

    #Flux
    Flux = AutoMPO()
    for n1=1: N
        for n2=1: N
            for i in QiNQiM(n1, n2, L)
                Flux += i
            end
        end
    end

    return MPO(Hopping + Mass + Electric + Flux, sites)
end

function calc_energy_gap(H, print)
    nsweeps = 15 # number of sweeps
    maxdim = [10,20,100,100,200, 2000] # gradually increase states kept
    cutoff = [1E-15] # desired truncation error

    state_array = [isodd(div(x-1, F*C) + 1) ? "Up" : "Dn" for x in 1:(N*F*C)]

    psi_init_0 = random_mps(sites, state_array)

    energy,psi0 = dmrg(H,psi_init_0;nsweeps,maxdim,cutoff, outputlevel=0)
    if print
        println("Generating ansatz for Excited State (Meson)...")
    end
    psi_init_1 = copy(psi0)
    successful_flip = false

    # Loop through every bond in the lattice to find a "flippable" pair
    for b in 1 : (N*F*C - 1)
        # Try direction 1
        op1 = op("S+", sites[b])
        op2 = op("S-", sites[b+1])
        
        psi_test = apply([op1, op2], psi_init_1; cutoff=1E-12)
        
        if norm(psi_test) > 0.1 # If norm is non-zero, flip is valid
            if print
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
            if print
                println("Found flippable ↑↓ site at bond $b. Creating Meson...")
            end
            psi_init_1 = psi_test
            successful_flip = true
            break
        end
    end

    if !successful_flip
        error("Could not find ANY spot to create a meson! The state might be fully polarized (ferromagnetic).")
    end

    normalize!(psi_init_1)

    # Now safe to run DMRG
    # weight can be smaller now that the state is physical
    energy1, psi1 = dmrg(H, [psi0], psi_init_1; nsweeps, cutoff, weight=100.0, outputlevel = 0)

    if print
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
  phase_diagram(25)
    # H = construct_hamiltonian(N, F, C, m0, a, g)

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
    
    #plot_observables(psi, N, F, C)

    #plot_correlations(psi)

    #plot_entanglement(psi, N*F*C)

end