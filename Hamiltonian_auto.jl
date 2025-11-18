using ITensors, ITensorMPS

m = 1
w = 1
N = 10
F = 1
C = 3
J = 1
sites = siteinds("S=1/2", (N*F*C))

function l(n, c)
    return ((n-1)*F*C) + c - 1
end

function addOp!(arr, op, site)
    push!(arr, op)
    push!(arr, site)
end

function addArr!(arr1, arr2)
    arr1[1] *= arr2[1]
    arr1 = append!(arr1, arr2[2:end])
end

function QijN(n, i, j, coeff)
    coeff *= 1/sqrt(2)
    ret = []
    push!(ret, coeff)
    addOp!(ret, "S+", l(n, i) + 1)
    addArr!(ret, BigLambda(n, i, j))
    addOp!(ret, "S-", l(n, j) + 1)
    mpo = AutoMPO()
    mpo += tuple(ret...)
    return MPO(mpo, sites)
end

function QiiN(n, i, coeff)
    coeff *= (1/(2*sqrt(2i*(i -1))))
    mpo = AutoMPO()
    for c=1 : i - 1
        mpo += coeff, "Z", l(n, c) + 1
        mpo += -1 * coeff, "Z", l(n, i) + 1
    end
    return MPO(mpo, sites)
end

function BigLambda(n, i, j)
    ret = []
    coeff = 1
    for k=l(n, j) : l(n, i) - 1
        coeff *= -1
        addOp!(ret, "Z", k + 1)
    end
    pushfirst!(ret, coeff)
    return ret
end 

function QiNQiM(n, m, coeff)
    ret = MPO()
    flag = true
    for i=2 : C 
        for j=1: i - 1
            A = QijN(n, j, i, coeff)
            B = QijN(m, i, j, coeff)
            mpo1 = contract(A, B; alg="naive", truncate=false)
            mpo1 = apply(A, B; alg="naive", truncate=false)
            if flag
                ret = mpo1
                flag = false
            else 
                +(ret, mpo1)
            end
            A = QijN(n, i, j, coeff)
            B = QijN(m, j, i, coeff)
            mpo2 = contract(A, B; alg="naive", truncate=false)
            mpo2 = apply(A, B; alg="naive", truncate=false)
            +(ret, mpo2)
        end
    end
    for i=2 : C
        A = QiiN(n, i, coeff)
        B = QiiN(m, i, coeff)
        mpo = contract(A, B; alg="naive", truncate=false)
        mpo = apply(A, B; alg="naive", truncate=false)
        +(ret, mpo)
    end
    return ret
end

let 

    #Mass
    function Mass()
        Mass = AutoMPO()
        for n=1: N 
            i = isodd(n) ? -1 : 1
            coeff = 0.5*m*i
            for c=1: C
                Mass += coeff, "Z", l(n, c) + 1
                Mass += coeff, "Id", l(n, c) + 1
            end
        end
        return MPO(Mass, sites)
    end 

    #Hopping
    function Hopping()
        Hopping = AutoMPO()
        for n=1: N-1
            for c=1: C
                for i=1 : 2
                    s1 = l(n+1, c)
                    s2 = l(n, c)
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
        return MPO(Hopping, sites)
    end

    #Electric
    function Electric()
        Electric = MPO()
        flag = true
        for n=1 : N - 1
            coeff = N - n
            if flag
                Electric = QiNQiM(n, n, J * coeff)
                flag = false
            else
                +(Electric, QiNQiM(n, n, J * coeff))
            end
        end
        for n=1 : N - 2
            for m=n+1 : N - 1
                coeff = N - m
                +(Electric, QiNQiM(n, m, 2 * J * coeff))
            end
        end
        return Electric
    end

    println(Hopping())
    println("------------------")
    println(Mass())
    println("------------------")
    println(Electric())
    H = +(Hopping() + Mass() + Electric())

    nsweeps = 5 # number of sweeps is 5
    maxdim = [10,20,100,100,200] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error

    psi0 = random_mps(sites;linkdims=2)

    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    print(energy)

end