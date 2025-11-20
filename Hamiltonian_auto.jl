using ITensors, ITensorMPS
using Plots

m = 2
w = 1
N = 20
F = 1
C = 3
J = 1
L = 10000
sites = siteinds("S=1/2", (N*F*C))

function l(n, f, c)
    return ((n-1)*F*C) + (C*(f -1)) + c - 1
end

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
            push!(ret, (coeff, "Z", l(n, f, c) + 1))
            push!(ret, (-1 * coeff, "Z", l(n, f, i) + 1))
        end
    end
    return ret
end

function BigLambda(n, f, i, j)
    ret = []
    coeff = 1
    for k=l(n, f, j) : l(n, f, i) - 1
        coeff *= -1
        addOp!(ret, "Z", k + 1)
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

function number_op(psi, w, n)
    num = AutoMPO()

    for c=1: C
        num += (w/2), "Sz", l(2*n, 1, c) + 1
        num += (w/2), "Sz", l((2*n)-1, 1, c) + 1
    end

    return inner(psi', MPO(num, siteinds(psi)), psi)
end

let 
    #Mass
    Mass = AutoMPO()
    for n=1: N 
        i = isodd(n) ? -1 : 1
        coeff = 0.5*m*i
        for f=1: F
            for c=1: C
                Mass += coeff, "Z", l(n, f, c) + 1
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
                            coeff *= -1
                            addOp!(temp, "Z", k + 1)
                        end
                        addOp!(temp, "S-", s2 + 1)
                        pushfirst!(temp, coeff)
                        Hopping += tuple(temp...)
                    else
                        temp = []
                        coeff = 1
                        addOp!(temp, "S+", s2 + 1)
                        for k=s2 : s1 - 1
                            coeff *= -1
                            addOp!(temp, "Z", k + 1)
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

    H = MPO(Hopping + Mass + Electric + Flux, sites)

    nsweeps = 50 # number of sweeps is 5
    maxdim = [10,20,100,100,200] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error

    psi0 = random_mps(sites;linkdims=2)

    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    

    wfs = [psi0]

    energy1, psi1 = dmrg(H, wfs, random_mps(sites;linkdims=2); nsweeps, cutoff, weight= 20.0)

    print("Ground State Energy = ")
    println(energy)
    print("Excited State Energy = ")
    println(energy1)
    print("Energy Gap = ")
    println(energy1 - energy)

    #plot_observables(psi, N, F, C)

    #plot_correlations(psi)

    #plot_entanglement(psi, N*F*C)
    ret = []
    for n=1: div(N,2)
        push!(ret, number_op(psi0, w, n))
    end

    p = plot(title="Number operator vaules for Random and Minimized states", xlabel="Site", ylabel="Number operator value")
    plot!(p, ret, label="Minimized")
    display(p)

end