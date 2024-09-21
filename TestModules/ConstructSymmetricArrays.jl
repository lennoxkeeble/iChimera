#= 
    In this module we construct totally symmetric rank 2, 3, and 4 tensors (arrays) with indices which run from 1, 2, 3. In these @views functions,
    the tensors are @views functions of time, so each element in the 3x3, 3x3x3, or 3x3x3x3 arrays will be a vector, but the type declarations
    can easily be amended to change the use of these @views functions.
=#

module ConstructSymmetricArrays

# Note that a note that a totally-symmetric tensor with r indices, each running from 1,...,d has (d-1+r)!/((d-1)! r!) independent components. We therefore need only 
# specify 6, 10, and 15 components of totally symmetric tensors with d=3 and r = 2, 3, and 4, respectively. Below we specify the indices of the independent components
# we choose to input to construct for each tensor
const two_index_components::Vector{Tuple{Int64, Int64}} = [(1, 2), (1, 3), (2, 3), (1, 1), (2, 2), (3, 3)];
const three_index_components::Vector{Tuple{Int64, Int64, Int64}} = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 3, 3), 
                                                                    (1, 2, 3), (2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3)];
const four_index_components::Vector{Tuple{Int64, Int64, Int64, Int64}} = [(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2), (1, 1, 1, 3), (1, 1, 3, 3),
                                                                          (1, 3, 3, 3), (1, 1, 2, 3), (1, 2, 2, 3), (1, 2, 3, 3), (2, 2, 2, 2), (2, 2, 2, 3),
                                                                          (2, 2, 3, 3), (2, 3, 3, 3), (3, 3, 3, 3)];
const traj_indices::Vector{Tuple} = [(1, 2), (1, 3), (2, 3), (1, 1), (2, 2), (3, 3), 
                                                   (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 3, 3), 
                                                   (1, 2, 3), (2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3)]
const waveform_indices::Vector{Tuple} = [(1, 2), (1, 3), (2, 3), (1, 1), (2, 2), (3, 3), 
                                         (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 3, 3), 
                                         (1, 2, 3), (2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3), 
                                         (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2), (1, 1, 1, 3), (1, 1, 3, 3),
                                         (1, 3, 3, 3), (1, 1, 2, 3), (1, 2, 2, 3), (1, 2, 3, 3), (2, 2, 2, 2), (2, 2, 2, 3),
                                         (2, 2, 3, 3), (2, 3, 3, 3), (3, 3, 3, 3)]
                                                

# constructs totally symmetric two-index tensor from the specified six independent components
@views function ConstructTwoIndexTensor(A11::Vector{Float64}, A12::Vector{Float64}, A13::Vector{Float64}, A22::Vector{Float64}, 
    A23::Vector{Float64}, A33::Vector{Float64})
    A = [Float64[] for i=1:3, j=1:3]
    @inbounds for (i, j) in two_index_components
        if i == 1 && j == 1
            A[i, j] = A11
        elseif i == 1 && j == 2
            A[i, j] = A12
            A[j, i] = A12
        elseif i == 1 && j == 3
            A[i, j] = A13
            A[j, i] = A13
        elseif i == 2 && j == 2
            A[i, j] = A22
        elseif i == 2 && j == 3
            A[i, j] = A23
            A[j, i] = A23
        elseif i == 3 && j == 3
            A[i, j] = A33
        end    
    end
    return A
end

# symmetrizes a two-index tensor with its six independent components already specified
@views function SymmetrizeTwoIndexTensor!(A::AbstractArray)
    @inbounds for (i, j) in two_index_components
        if i == 1 && j == 2
            A[2, 1] = A[1, 2]
        elseif i == 1 && j == 3
            A[3, 1] = A[1, 3]
        elseif i == 2 && j == 3
            A[3, 2] = A[2, 3]
        end    
    end
end


# constructs totally symmetric three-index tensor from the specified six independent components
@views function ConstructThreeIndexTensor(A111::Vector{Float64}, A112::Vector{Float64}, A122::Vector{Float64}, A113::Vector{Float64}, A133::Vector{Float64}, 
    A123::Vector{Float64}, A222::Vector{Float64}, A223::Vector{Float64}, A233::Vector{Float64}, A333::Vector{Float64})
    A = [Float64[] for i=1:3, j=1:3, k=1:3]
    @inbounds for (i, j, k) in three_index_components
        if i == 1 && j == 1 && k == 1
            A[1, 1, 1] = A111;
        elseif i == 1 && j == 1 && k == 2 
            A[1, 1, 2] = A112;
            A[1, 2, 1] = A112;
            A[2, 1, 1] = A112;
        elseif i == 1 && j == 2 && k == 2 
            A[1, 2, 2] = A122;
            A[2, 1, 2] = A122;
            A[2, 2, 1] = A122;
        elseif i == 1 && j == 1 && k == 3 
            A[1, 1, 3] = A113;
            A[1, 3, 1] = A113;
            A[3, 1, 1] = A113;
        elseif i == 1 && j == 3 && k == 3
            A[1, 3, 3] = A133;
            A[3, 1, 3] = A133;
            A[3, 3, 1] = A133;
        elseif i == 1 && j == 2 && k == 3 
            A[1, 2, 3] = A123;
            A[1, 3, 2] = A123;
            A[2, 1, 3] = A123;
            A[2, 3, 1] = A123;
            A[3, 2, 1] = A123;
            A[3, 1, 2] = A123;
        elseif i == 2 && j == 3 && k == 3 
            A[2, 3, 3] = A233;
            A[3, 2, 3] = A233;
            A[3, 3, 2] = A233;
        elseif i == 2 && j == 2 && k == 3 
            A[2, 2, 3] = A223;
            A[2, 3, 2] = A223;
            A[3, 2, 2] = A223;
        elseif i == 2 && j == 2 && k == 2
            A[2, 2, 2] = A222
        elseif i == 3 && j == 3 && k == 3
            A[3, 3, 3] = A333   
        end
    end
    return A
end

# symmetrizes a three-index tensor with its ten independent components already specified
@views function SymmetrizeThreeIndexTensor!(A::AbstractArray)
    @inbounds for (i, j, k) in three_index_components
        if i == 1 && j == 1 && k == 2 
            A[1, 2, 1] = A[1, 1, 2];
            A[2, 1, 1] = A[1, 1, 2];
        elseif i == 1 && j == 2 && k == 2 
            A[2, 1, 2] = A[1, 2, 2];
            A[2, 2, 1] = A[1, 2, 2];
        elseif i == 1 && j == 1 && k == 3 
            A[1, 3, 1] = A[1, 1, 3];
            A[3, 1, 1] = A[1, 1, 3];
        elseif i == 1 && j == 3 && k == 3
            A[3, 1, 3] = A[1, 3, 3];
            A[3, 3, 1] = A[1, 3, 3];
        elseif i == 1 && j == 2 && k == 3 
            A[1, 3, 2] = A[1, 2, 3];
            A[2, 1, 3] = A[1, 2, 3];
            A[2, 3, 1] = A[1, 2, 3];
            A[3, 2, 1] = A[1, 2, 3];
            A[3, 1, 2] = A[1, 2, 3];
        elseif i == 2 && j == 3 && k == 3 
            A[3, 2, 3] = A[2, 3, 3];
            A[3, 3, 2] = A[2, 3, 3];
        elseif i == 2 && j == 2 && k == 3 
            A[2, 3, 2] = A[2, 2, 3];
            A[3, 2, 2] = A[2, 2, 3];
        end
    end
end

# constructs totally symmetric four-index tensor from the specified fifteen independent components
@views function ConstructFourIndexTensor(A1111::Vector{Float64}, A1112::Vector{Float64}, A1122::Vector{Float64}, A1222::Vector{Float64}, A1113::Vector{Float64}, 
    A1133::Vector{Float64}, A1333::Vector{Float64}, A1123::Vector{Float64}, A1223::Vector{Float64}, A1233::Vector{Float64}, A2222::Vector{Float64}, 
    A2223::Vector{Float64}, A2233::Vector{Float64}, A2333::Vector{Float64}, A3333::Vector{Float64})
    A = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
    @inbounds for (i, j, k, l) in four_index_components
        if i == 1 && j == 1 && k == 1 && l == 1
            A[1, 1, 1, 1] = A1111
        elseif i == 1 && j == 1 && k == 1 && l == 2 
            A[1, 1, 1, 2] = A1112;
            A[1, 1, 2, 1] = A1112;
            A[1, 2, 1, 1] = A1112;
            A[2, 1, 1, 1] = A1112;
        elseif i == 1 && j == 1 && k == 2 && l == 2
            A[1, 1, 2, 2] = A1122;
            A[1, 2, 1, 2] = A1122;
            A[1, 2, 2, 1] = A1122;
            A[2, 1, 1, 2] = A1122;
            A[2, 1, 2, 1] = A1122;
            A[2, 2, 1, 1] = A1122;
        elseif i == 1 && j == 2 && k == 2 && l == 2 
            A[1, 2, 2, 2] = A1222;
            A[2, 1, 2, 2] = A1222;
            A[2, 2, 1, 2] = A1222;
            A[2, 2, 2, 1] = A1222;
        elseif i == 1 && j == 1 && k == 1 && l == 3 
            A[1, 1, 1, 3] = A1113;
            A[1, 1, 3, 1] = A1113;
            A[1, 3, 1, 1] = A1113;
            A[3, 1, 1, 1] = A1113;
        elseif i == 1 && j == 1 && k == 3 && l == 3
            A[1, 1, 3, 3] = A1133;
            A[1, 3, 1, 3] = A1133;
            A[1, 3, 3, 1] = A1133;
            A[3, 1, 1, 3] = A1133;
            A[3, 1, 3, 1] = A1133;
            A[3, 3, 1, 1] = A1133;
        elseif i == 1 && j == 3 && k == 3 && l == 3 
            A[1, 3, 3, 3] = A1333;
            A[3, 1, 3, 3] = A1333;
            A[3, 3, 1, 3] = A1333;
            A[3, 3, 3, 1] = A1333;
        elseif i == 1 && j == 1 && k == 2 && l == 3
            A[1, 1, 2, 3] = A1123;
            A[1, 2, 1, 3] = A1123;
            A[1, 2, 3, 1] = A1123;
            A[1, 3, 1, 2] = A1123;
            A[1, 3, 2, 1] = A1123;
            A[2, 1, 1, 3] = A1123;
            A[2, 1, 3, 1] = A1123;
            A[2, 3, 1, 1] = A1123;
            A[3, 1, 1, 2] = A1123;
            A[3, 1, 2, 1] = A1123;
            A[3, 2, 1, 1] = A1123;
            A[1, 1, 3, 2] = A1123;
        elseif i == 1 && j == 2 && k == 2 && l == 3
            A[1, 2, 2, 3] = A1223;
            A[1, 2, 3, 2] = A1223;
            A[1, 3, 2, 2] = A1223;
            A[2, 2, 1, 3] = A1223;
            A[2, 2, 3, 1] = A1223;
            A[2, 1, 2, 3] = A1223;
            A[2, 1, 3, 2] = A1223;
            A[2, 3, 1, 2] = A1223;
            A[2, 3, 2, 1] = A1223;
            A[3, 2, 2, 1] = A1223;
            A[3, 2, 1, 2] = A1223;
            A[3, 1, 2, 2] = A1223;
        elseif i == 1 && j == 2 && k == 3 && l == 3
            A[1, 2, 3, 3] = A1233;
            A[1, 3, 2, 3] = A1233;
            A[1, 3, 3, 2] = A1233;
            A[2, 1, 3, 3] = A1233;
            A[2, 3, 1, 3] = A1233;
            A[2, 3, 3, 1] = A1233;
            A[3, 1, 2, 3] = A1233;
            A[3, 1, 3, 2] = A1233;
            A[3, 2, 1, 3] = A1233;
            A[3, 2, 3, 1] = A1233;
            A[3, 3, 1, 2] = A1233;
            A[3, 3, 2, 1] = A1233;
        elseif i == 2 && j == 2 && k == 2 && l == 2
            A[2, 2, 2, 2] = A2222
        elseif i == 2 && j == 2 && k == 2 && l == 3 
            A[2, 2, 2, 3] = A2223;
            A[2, 2, 3, 2] = A2223;
            A[2, 3, 2, 2] = A2223;
            A[3, 2, 2, 2] = A2223;
        elseif i == 2 && j == 2 && k == 3 && l == 3
            A[2, 2, 3, 3] = A2233;
            A[2, 3, 2, 3] = A2233;
            A[2, 3, 3, 2] = A2233;
            A[3, 2, 2, 3] = A2233;
            A[3, 2, 3, 2] = A2233;
            A[3, 3, 2, 2] = A2233;
        elseif i == 2 && j == 3 && k == 3 && l == 3 
            A[2, 3, 3, 3] = A2333;
            A[3, 2, 3, 3] = A2333;
            A[3, 3, 2, 3] = A2333;
            A[3, 3, 3, 2] = A2333;
        elseif i == 3 && j == 3 && k == 3 && l == 3
            A[3, 3, 3, 3] = A3333
        end
    end
    return A
end


# symmetrizes a four-index tensor with its fifteen independent components already specified
@views function SymmetrizeFourIndexTensor!(A::AbstractArray)
    @inbounds for (i, j, k, l) in four_index_components
        if i == 1 && j == 1 && k == 1 && l == 2 
            A[1, 1, 2, 1] = A[1, 1, 1, 2];
            A[1, 2, 1, 1] = A[1, 1, 1, 2];
            A[2, 1, 1, 1] = A[1, 1, 1, 2];
        elseif i == 1 && j == 1 && k == 2 && l == 2
            A[1, 2, 1, 2] = A[1, 1, 2, 2];
            A[1, 2, 2, 1] = A[1, 1, 2, 2];
            A[2, 1, 1, 2] = A[1, 1, 2, 2];
            A[2, 1, 2, 1] = A[1, 1, 2, 2];
            A[2, 2, 1, 1] = A[1, 1, 2, 2];
        elseif i == 1 && j == 2 && k == 2 && l == 2 
            A[2, 1, 2, 2] = A[1, 2, 2, 2];
            A[2, 2, 1, 2] = A[1, 2, 2, 2];
            A[2, 2, 2, 1] = A[1, 2, 2, 2];
        elseif i == 1 && j == 1 && k == 1 && l == 3 
            A[1, 1, 3, 1] = A[1, 1, 1, 3];
            A[1, 3, 1, 1] = A[1, 1, 1, 3];
            A[3, 1, 1, 1] = A[1, 1, 1, 3];
        elseif i == 1 && j == 1 && k == 3 && l == 3
            A[1, 3, 1, 3] = A[1, 1, 3, 3];
            A[1, 3, 3, 1] = A[1, 1, 3, 3];
            A[3, 1, 1, 3] = A[1, 1, 3, 3];
            A[3, 1, 3, 1] = A[1, 1, 3, 3];
            A[3, 3, 1, 1] = A[1, 1, 3, 3];
        elseif i == 1 && j == 3 && k == 3 && l == 3 
            A[3, 1, 3, 3] = A[1, 3, 3, 3];
            A[3, 3, 1, 3] = A[1, 3, 3, 3];
            A[3, 3, 3, 1] = A[1, 3, 3, 3];
        elseif i == 1 && j == 1 && k == 2 && l == 3
            A[1, 2, 1, 3] = A[1, 1, 2, 3];
            A[1, 2, 3, 1] = A[1, 1, 2, 3];
            A[1, 3, 1, 2] = A[1, 1, 2, 3];
            A[1, 3, 2, 1] = A[1, 1, 2, 3];
            A[2, 1, 1, 3] = A[1, 1, 2, 3];
            A[2, 1, 3, 1] = A[1, 1, 2, 3];
            A[2, 3, 1, 1] = A[1, 1, 2, 3];
            A[3, 1, 1, 2] = A[1, 1, 2, 3];
            A[3, 1, 2, 1] = A[1, 1, 2, 3];
            A[3, 2, 1, 1] = A[1, 1, 2, 3];
            A[1, 1, 3, 2] = A[1, 1, 2, 3];
        elseif i == 1 && j == 2 && k == 2 && l == 3
            A[1, 2, 3, 2] = A[1, 2, 2, 3];
            A[1, 3, 2, 2] = A[1, 2, 2, 3];
            A[2, 2, 1, 3] = A[1, 2, 2, 3];
            A[2, 2, 3, 1] = A[1, 2, 2, 3];
            A[2, 1, 2, 3] = A[1, 2, 2, 3];
            A[2, 1, 3, 2] = A[1, 2, 2, 3];
            A[2, 3, 1, 2] = A[1, 2, 2, 3];
            A[2, 3, 2, 1] = A[1, 2, 2, 3];
            A[3, 2, 2, 1] = A[1, 2, 2, 3];
            A[3, 2, 1, 2] = A[1, 2, 2, 3];
            A[3, 1, 2, 2] = A[1, 2, 2, 3];
        elseif i == 1 && j == 2 && k == 3 && l == 3
            A[1, 3, 2, 3] = A[1, 2, 3, 3];
            A[1, 3, 3, 2] = A[1, 2, 3, 3];
            A[2, 1, 3, 3] = A[1, 2, 3, 3];
            A[2, 3, 1, 3] = A[1, 2, 3, 3];
            A[2, 3, 3, 1] = A[1, 2, 3, 3];
            A[3, 1, 2, 3] = A[1, 2, 3, 3];
            A[3, 1, 3, 2] = A[1, 2, 3, 3];
            A[3, 2, 1, 3] = A[1, 2, 3, 3];
            A[3, 2, 3, 1] = A[1, 2, 3, 3];
            A[3, 3, 1, 2] = A[1, 2, 3, 3];
            A[3, 3, 2, 1] = A[1, 2, 3, 3];
        elseif i == 2 && j == 2 && k == 2 && l == 3 
            A[2, 2, 3, 2] = A[2, 2, 2, 3];
            A[2, 3, 2, 2] = A[2, 2, 2, 3];
            A[3, 2, 2, 2] = A[2, 2, 2, 3];
        elseif i == 2 && j == 2 && k == 3 && l == 3
            A[2, 3, 2, 3] = A[2, 2, 3, 3];
            A[2, 3, 3, 2] = A[2, 2, 3, 3];
            A[3, 2, 2, 3] = A[2, 2, 3, 3];
            A[3, 2, 3, 2] = A[2, 2, 3, 3];
            A[3, 3, 2, 2] = A[2, 2, 3, 3];
        elseif i == 2 && j == 3 && k == 3 && l == 3 
            A[3, 2, 3, 3] = A[2, 3, 3, 3];
            A[3, 3, 2, 3] = A[2, 3, 3, 3];
            A[3, 3, 3, 2] = A[2, 3, 3, 3];
        end
    end
end

end