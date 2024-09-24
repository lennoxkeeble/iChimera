module FiniteDiff_4

# given an array length and an index, this function returns the node of the index
function compute_node(index::Int64, array_len::Int64, node_max::Int64)
    num_left = index-1     # number of array entries to the left
    num_right = array_len-index    # number of array entries to the right
    min_distance=minimum([num_left, num_right])    # minimum "distance" to one either end of the array

    # if maxiumum node is even
    if iseven(node_max)
        # only if there are less than (node_max/2-1) points to the left or right must we change the node order
        if min_distance<(node_max/2 - 1)
            # return node value based on whether we are on the left or right
            return num_left < num_right ? num_left+1 : node_max - num_right
        else
            return num_left < num_right ? node_max÷2 : node_max÷2 + 1    # this (perhaps inconsequentially) ensures 'symmetry'
        end
    # maximum node is odd
    else
        # only if there are less than (node_max ÷ 2) points to the left or right must we change the node order
        if min_distance<(node_max ÷ 2)
            # return node value based on whether we are on the left or right
            return num_left < num_right ? num_left+1 : node_max - num_right
        else
            return (node_max ÷ 2) + 1
        end
    end
end

# first derivative
deriv_1_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(-25*y[i]+48*y[1+i]-36*y[2+i]+16*y[3+i]-3*y[4+i])/(12*h)
deriv_1_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(-3*y[-1+i]-10*y[i]+18*y[1+i]-6*y[2+i]+y[3+i])/(12*h)
deriv_1_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(y[-2+i]-8*y[-1+i]+8*y[1+i]-y[2+i])/(12*h)
deriv_1_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-3+i]+6*y[-2+i]-18*y[-1+i]+10*y[i]+3*y[1+i])/(12*h)
deriv_1_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(3*y[-4+i]-16*y[-3+i]+36*y[-2+i]-48*y[-1+i]+25*y[i])/(12*h)

# depending on the node, compute the first derivative
compute_first_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_1_node_1(y,h,i) : node==2 ? deriv_1_node_2(y,h,i) : node==3 ? deriv_1_node_3(y,h,i) : node==4 ? deriv_1_node_4(y,h,i) : node==5 ? deriv_1_node_5(y,h,i) : throw(DomainError(node, "node must be ≤ 5"))

# compute first derivative of points in array y
function compute_first_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64},h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_first_derivative(y,h, i, compute_node(i, nPoints, 5))
    end
end

# second derivative
deriv_2_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(45*y[i]-154*y[1+i]+214*y[2+i]-156*y[3+i]+61*y[4+i]-10*y[5+i])/(12*h^2)
deriv_2_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(10*y[-1+i]-15*y[i]-4*y[1+i]+14*y[2+i]-6*y[3+i]+y[4+i])/(12*h^2)
deriv_2_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-2+i]+16*y[-1+i]-30*y[i]+16*y[1+i]-y[2+i])/(12*h^2)
deriv_2_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-2+i]+16*y[-1+i]-30*y[i]+16*y[1+i]-y[2+i])/(12*h^2)
deriv_2_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(y[-4+i]-6*y[-3+i]+14*y[-2+i]-4*y[-1+i]-15*y[i]+10*y[1+i])/(12*h^2)
deriv_2_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(-10*y[-5+i]+61*y[-4+i]-156*y[-3+i]+214*y[-2+i]-154*y[-1+i]+45*y[i])/(12*h^2)

# depending on the node, compute the second derivative
compute_second_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_2_node_1(y,h,i) : node==2 ? deriv_2_node_2(y,h,i) : node==3 ? deriv_2_node_3(y,h,i) : node==4 ? deriv_2_node_4(y,h,i) : node==5 ? deriv_2_node_5(y,h,i) : node==6 ? deriv_2_node_6(y,h,i) : throw(DomainError(node, "node must be ≤ 6"))

# compute second derivative of points in array y
function compute_second_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64},h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_second_derivative(y,h, i, compute_node(i, nPoints, 6))
    end
end

# third derivative
deriv_3_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(-49*y[i]+232*y[1+i]-461*y[2+i]+496*y[3+i]-307*y[4+i]+104*y[5+i]-15*y[6+i])/(8*h^3)
deriv_3_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(-15*y[-1+i]+56*y[i]-83*y[1+i]+64*y[2+i]-29*y[3+i]+8*y[4+i]-y[5+i])/(8*h^3)
deriv_3_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-2+i]-8*y[-1+i]+35*y[i]-48*y[1+i]+29*y[2+i]-8*y[3+i]+y[4+i])/(8*h^3)
deriv_3_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(y[-3+i]-8*y[-2+i]+13*y[-1+i]-13*y[1+i]+8*y[2+i]-y[3+i])/(8*h^3)
deriv_3_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-4+i]+8*y[-3+i]-29*y[-2+i]+48*y[-1+i]-35*y[i]+8*y[1+i]+y[2+i])/(8*h^3)
deriv_3_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(y[-5+i]-8*y[-4+i]+29*y[-3+i]-64*y[-2+i]+83*y[-1+i]-56*y[i]+15*y[1+i])/(8*h^3)
deriv_3_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(15*y[-6+i]-104*y[-5+i]+307*y[-4+i]-496*y[-3+i]+461*y[-2+i]-232*y[-1+i]+49*y[i])/(8*h^3)

# depending on the node, compute the third derivative
compute_third_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_3_node_1(y,h,i) : node==2 ? deriv_3_node_2(y,h,i) : node==3 ? deriv_3_node_3(y,h,i) : node==4 ? deriv_3_node_4(y,h,i) : node==5 ? deriv_3_node_5(y,h,i) : node==6 ? deriv_3_node_6(y,h,i) : node==7 ? deriv_3_node_7(y,h,i) : throw(DomainError(node, "node must be ≤ 7"))

# compute third derivative of points in array y
function compute_third_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64},h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_third_derivative(y,h, i, compute_node(i, nPoints, 7))
    end
end

# fourth derivative
deriv_4_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(56*y[i]-333*y[1+i]+852*y[2+i]-1219*y[3+i]+1056*y[4+i]-555*y[5+i]+164*y[6+i]-21*y[7+i])/(6*h^4)
deriv_4_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(21*y[-1+i]-112*y[i]+255*y[1+i]-324*y[2+i]+251*y[3+i]-120*y[4+i]+33*y[5+i]-4*y[6+i])/(6*h^4)
deriv_4_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(4*y[-2+i]-11*y[-1+i]+31*y[1+i]-44*y[2+i]+27*y[3+i]-8*y[4+i]+y[5+i])/(6*h^4)
deriv_4_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-3+i]+12*y[-2+i]-39*y[-1+i]+56*y[i]-39*y[1+i]+12*y[2+i]-y[3+i])/(6*h^4)
deriv_4_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-3+i]+12*y[-2+i]-39*y[-1+i]+56*y[i]-39*y[1+i]+12*y[2+i]-y[3+i])/(6*h^4)
deriv_4_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(y[-5+i]-8*y[-4+i]+27*y[-3+i]-44*y[-2+i]+31*y[-1+i]-11*y[1+i]+4*y[2+i])/(6*h^4)
deriv_4_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(-4*y[-6+i]+33*y[-5+i]-120*y[-4+i]+251*y[-3+i]-324*y[-2+i]+255*y[-1+i]-112*y[i]+21*y[1+i])/(6*h^4)
deriv_4_node_8(y::AbstractVector{Float64},h::Float64,i::Int64)=(-21*y[-7+i]+164*y[-6+i]-555*y[-5+i]+1056*y[-4+i]-1219*y[-3+i]+852*y[-2+i]-333*y[-1+i]+56*y[i])/(6*h^4)

# depending on the node, compute the fourth derivative
compute_fourth_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_4_node_1(y,h,i) : node==2 ? deriv_4_node_2(y,h,i) : node==3 ? deriv_4_node_3(y,h,i) : node==4 ? deriv_4_node_4(y,h,i) : node==5 ? deriv_4_node_5(y,h,i) : node==6 ? deriv_4_node_6(y,h,i) : node==7 ? deriv_4_node_7(y,h,i) : node==8 ? deriv_4_node_8(y,h,i) : throw(DomainError(node, "node must be ≤ 8"))

# compute fourth derivative of points in array y
function compute_fourth_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64},h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_fourth_derivative(y,h, i, compute_node(i, nPoints, 8))
    end
end

# fifth derivative
deriv_5_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(-81*y[i]+575*y[1+i]-1790*y[2+i]+3195*y[3+i]-3580*y[4+i]+2581*y[5+i]-1170*y[6+i]+305*y[7+i]-35*y[8+i])/(6*h^5)
deriv_5_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(-35*y[-1+i]+234*y[i]-685*y[1+i]+1150*y[2+i]-1215*y[3+i]+830*y[4+i]-359*y[5+i]+90*y[6+i]-10*y[7+i])/(6*h^5)
deriv_5_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(-10*y[-2+i]+55*y[-1+i]-126*y[i]+155*y[1+i]-110*y[2+i]+45*y[3+i]-10*y[4+i]+y[5+i])/(6*h^5)    ### NOT SURE ABOUT THIS - WHY IS THIS EXPRESSION THE SAME AS THE ONE BELOW?
deriv_5_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(-10*y[-2+i]+55*y[-1+i]-126*y[i]+155*y[1+i]-110*y[2+i]+45*y[3+i]-10*y[4+i]+y[5+i])/(6*h^5)
deriv_5_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(y[-4+i]-9*y[-3+i]+26*y[-2+i]-29*y[-1+i]+29*y[1+i]-26*y[2+i]+9*y[3+i]-y[4+i])/(6*h^5)
deriv_5_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-5+i]+10*y[-4+i]-45*y[-3+i]+110*y[-2+i]-155*y[-1+i]+126*y[i]-55*y[1+i]+10*y[2+i])/(6*h^5)
deriv_5_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-5+i]+10*y[-4+i]-45*y[-3+i]+110*y[-2+i]-155*y[-1+i]+126*y[i]-55*y[1+i]+10*y[2+i])/(6*h^5)
deriv_5_node_8(y::AbstractVector{Float64},h::Float64,i::Int64)=(10*y[-7+i]-90*y[-6+i]+359*y[-5+i]-830*y[-4+i]+1215*y[-3+i]-1150*y[-2+i]+685*y[-1+i]-234*y[i]+35*y[1+i])/(6*h^5)
deriv_5_node_9(y::AbstractVector{Float64},h::Float64,i::Int64)=(35*y[-8+i]-305*y[-7+i]+1170*y[-6+i]-2581*y[-5+i]+3580*y[-4+i]-3195*y[-3+i]+1790*y[-2+i]-575*y[-1+i]+81*y[i])/(6*h^5)

# depending on the node, compute the fifth derivative
compute_fifth_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_5_node_1(y,h,i) : node==2 ? deriv_5_node_2(y,h,i) : node==3 ? deriv_5_node_3(y,h,i) : node==4 ? deriv_5_node_4(y,h,i) : node==5 ? deriv_5_node_5(y,h,i) : node==6 ? deriv_5_node_6(y,h,i) : node==7 ? deriv_5_node_7(y,h,i) : node==8 ? deriv_5_node_8(y,h,i) : node==9 ? deriv_5_node_9(y,h,i) : throw(DomainError(node, "node must be ≤ 9"))

# compute fifth derivative of points in array y
function compute_fifth_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64},h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_fifth_derivative(y,h, i, compute_node(i, nPoints, 9))
    end
end


# sixth derivative
deriv_6_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(75*y[i]-616*y[1+i]+2252*y[2+i]-4812*y[3+i]+6626*y[4+i]-6100*y[5+i]+3756*y[6+i]-1492*y[7+i]+347*y[8+i]-36*y[9+i])/(4*h^6)
deriv_6_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(36*y[-1+i]-285*y[i]+1004*y[1+i]-2068*y[2+i]+2748*y[3+i]-2446*y[4+i]+1460*y[5+i]-564*y[6+i]+128*y[7+i]-13*y[8+i])/(4*h^6)
deriv_6_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(13*y[-2+i]-94*y[-1+i]+300*y[i]-556*y[1+i]+662*y[2+i]-528*y[3+i]+284*y[4+i]-100*y[5+i]+21*y[6+i]-2*y[7+i])/(4*h^6)
deriv_6_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(2*y[-3+i]-7*y[-2+i]-4*y[-1+i]+60*y[i]-136*y[1+i]+158*y[2+i]-108*y[3+i]+44*y[4+i]-10*y[5+i]+y[6+i])/(4*h^6)
deriv_6_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-4+i]+12*y[-3+i]-52*y[-2+i]+116*y[-1+i]-150*y[i]+116*y[1+i]-52*y[2+i]+12*y[3+i]-y[4+i])/(4*h^6)
deriv_6_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(-y[-4+i]+12*y[-3+i]-52*y[-2+i]+116*y[-1+i]-150*y[i]+116*y[1+i]-52*y[2+i]+12*y[3+i]-y[4+i])/(4*h^6)
deriv_6_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(y[-6+i]-10*y[-5+i]+44*y[-4+i]-108*y[-3+i]+158*y[-2+i]-136*y[-1+i]+60*y[i]-4*y[1+i]-7*y[2+i]+2*y[3+i])/(4*h^6)
deriv_6_node_8(y::AbstractVector{Float64},h::Float64,i::Int64)=(-2*y[-7+i]+21*y[-6+i]-100*y[-5+i]+284*y[-4+i]-528*y[-3+i]+662*y[-2+i]-556*y[-1+i]+300*y[i]-94*y[1+i]+13*y[2+i])/(4*h^6)
deriv_6_node_9(y::AbstractVector{Float64},h::Float64,i::Int64)=(-13*y[-8+i]+128*y[-7+i]-564*y[-6+i]+1460*y[-5+i]-2446*y[-4+i]+2748*y[-3+i]-2068*y[-2+i]+1004*y[-1+i]-285*y[i]+36*y[1+i])/(4*h^6)
deriv_6_node_10(y::AbstractVector{Float64},h::Float64,i::Int64)=(-36*y[-9+i]+347*y[-8+i]-1492*y[-7+i]+3756*y[-6+i]-6100*y[-5+i]+6626*y[-4+i]-4812*y[-3+i]+2252*y[-2+i]-616*y[-1+i]+75*y[i])/(4*h^6)

# depending on the node, compute the sixth derivative
compute_sixth_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_6_node_1(y,h,i) : node==2 ? deriv_6_node_2(y,h,i) : node==3 ? deriv_6_node_3(y,h,i) : node==4 ? deriv_6_node_4(y,h,i) : node==5 ? deriv_6_node_5(y,h,i) : node==6 ? deriv_6_node_6(y,h,i) : node==7 ? deriv_6_node_7(y,h,i) : node==8 ? deriv_6_node_8(y,h,i) : node==9 ? deriv_6_node_9(y,h,i) : node==10 ? deriv_6_node_10(y,h,i) : throw(DomainError(node, "node must be ≤ 10"))

# compute sixth derivative of points in array y
function compute_sixth_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64},h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_sixth_derivative(y,h, i, compute_node(i, nPoints, 10))
    end
end

function compute_derivs(derivs::AbstractArray, ydata::AbstractVector{Float64},h::Float64, nPoints::Int64)
    compute_first_derivative(derivs[1], ydata,h, nPoints)
    compute_second_derivative(derivs[2], ydata,h, nPoints)
    # compute_third_derivative(derivs[3], ydata,h, nPoints)
    # compute_fourth_derivative(derivs[4], ydata,h, nPoints)
    # compute_fifth_derivative(derivs[5], ydata,h, nPoints)
    # compute_sixth_derivative(derivs[6], ydata,h, nPoints)
end

end

module FiniteDiff_5

# given an array length and an index, this function returns the node of the index
function compute_node(index::Int64, array_len::Int64, node_max::Int64)
    num_left = index-1     # number of array entries to the left
    num_right = array_len-index    # number of array entries to the right
    min_distance=minimum([num_left, num_right])    # minimum "distance" to one either end of the array

    # if maxiumum node is even
    if iseven(node_max)
        # only if there are less than (node_max/2-1) points to the left or right must we change the node order
        if min_distance<(node_max/2 - 1)
            # return node value based on whether we are on the left or right
            return num_left < num_right ? num_left+1 : node_max - num_right
        else
            return num_left < num_right ? node_max÷2 : node_max÷2 + 1    # this (perhaps inconsequentially) ensures 'symmetry'
        end
    # maximum node is odd
    else
        # only if there are less than (node_max ÷ 2) points to the left or right must we change the node order
        if min_distance<(node_max ÷ 2)
            # return node value based on whether we are on the left or right
            return num_left < num_right ? num_left+1 : node_max - num_right
        else
            return (node_max ÷ 2) + 1
        end
    end
end

# first derivative
deriv_1_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(-137*y[i]+300*y[1+i]-300*y[2+i]+200*y[3+i]-75*y[4+i]+12*y[5+i])/(60*h)
deriv_1_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(-12*y[-1+i]-65*y[i]+120*y[1+i]-60*y[2+i]+20*y[3+i]-3*y[4+i])/(60*h)
deriv_1_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(3*y[-2+i]-30*y[-1+i]-20*y[i]+60*y[1+i]-15*y[2+i]+2*y[3+i])/(60*h)
deriv_1_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(-2*y[-3+i]+15*y[-2+i]-60*y[-1+i]+20*y[i]+30*y[1+i]-3*y[2+i])/(60*h)
deriv_1_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(3*y[-4+i]-20*y[-3+i]+60*y[-2+i]-120*y[-1+i]+65*y[i]+12*y[1+i])/(60*h)
deriv_1_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(-12*y[-5+i]+75*y[-4+i]-200*y[-3+i]+300*y[-2+i]-300*y[-1+i]+137*y[i])/(60*h)

# depending on the node, compute the first derivative
compute_first_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_1_node_1(y,h,i) : node==2 ? deriv_1_node_2(y,h,i) : node==3 ? deriv_1_node_3(y,h,i) : node==4 ? deriv_1_node_4(y,h,i) : node==5 ? deriv_1_node_5(y,h,i) : node==6 ? deriv_1_node_6(y,h,i) : throw(DomainError(node, "node must be ≤ 6"))

# compute first derivative of points in at index compute_at
function compute_first_derivative(compute_at::Int64, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    return compute_first_derivative(y, h, compute_at, compute_node(compute_at, nPoints, 6))
end

# compute first derivative of points in array y
function compute_first_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_first_derivative(y, h, i, compute_node(i, nPoints, 6))
    end
end

# second derivative
deriv_2_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(812*y[i]-3132*y[1+i]+5265*y[2+i]-5080*y[3+i]+2970*y[4+i]-972*y[5+i]+137*y[6+i])/(180*h^2)
deriv_2_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(137*y[-1+i]-147*y[i]-255*y[1+i]+470*y[2+i]-285*y[3+i]+93*y[4+i]-13*y[5+i])/(180*h^2)
deriv_2_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(-13*y[-2+i]+228*y[-1+i]-420*y[i]+200*y[1+i]+15*y[2+i]-12*y[3+i]+2*y[4+i])/(180*h^2)
deriv_2_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(2*y[-3+i]-27*y[-2+i]+270*y[-1+i]-490*y[i]+270*y[1+i]-27*y[2+i]+2*y[3+i])/(180*h^2)
deriv_2_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(2*y[-4+i]-12*y[-3+i]+15*y[-2+i]+200*y[-1+i]-420*y[i]+228*y[1+i]-13*y[2+i])/(180*h^2)
deriv_2_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(-13*y[-5+i]+93*y[-4+i]-285*y[-3+i]+470*y[-2+i]-255*y[-1+i]-147*y[i]+137*y[1+i])/(180*h^2)
deriv_2_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(137*y[-6+i]-972*y[-5+i]+2970*y[-4+i]-5080*y[-3+i]+5265*y[-2+i]-3132*y[-1+i]+812*y[i])/(180*h^2)

# depending on the node, compute the second derivative
compute_second_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_2_node_1(y,h,i) : node==2 ? deriv_2_node_2(y,h,i) : node==3 ? deriv_2_node_3(y,h,i) : node==4 ? deriv_2_node_4(y,h,i) : node==5 ? deriv_2_node_5(y,h,i) : node==6 ? deriv_2_node_6(y,h,i) : node==7 ? deriv_2_node_7(y,h,i) : throw(DomainError(node, "node must be ≤ 7"))

# compute second derivative of points in at index compute_at
function compute_second_derivative(compute_at::Int64, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    return compute_second_derivative(y, h, compute_at, compute_node(compute_at, nPoints, 7))
end

# compute second derivative of points in array y
function compute_second_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_second_derivative(y, h, i, compute_node(i, nPoints, 7))
    end
end

# third derivative
deriv_3_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(-967*y[i]+5104*y[1+i]-11787*y[2+i]+15560*y[3+i]-12725*y[4+i]+6432*y[5+i]-1849*y[6+i]+232*y[7+i])/(120*h^3)
deriv_3_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(-232*y[-1+i]+889*y[i]-1392*y[1+i]+1205*y[2+i]-680*y[3+i]+267*y[4+i]-64*y[5+i]+7*y[6+i])/(120*h^3)
deriv_3_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(-7*y[-2+i]-176*y[-1+i]+693*y[i]-1000*y[1+i]+715*y[2+i]-288*y[3+i]+71*y[4+i]-8*y[5+i])/(120*h^3)
deriv_3_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(8*y[-3+i]-71*y[-2+i]+48*y[-1+i]+245*y[i]-440*y[1+i]+267*y[2+i]-64*y[3+i]+7*y[4+i])/(120*h^3)
deriv_3_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(-7*y[-4+i]+64*y[-3+i]-267*y[-2+i]+440*y[-1+i]-245*y[i]-48*y[1+i]+71*y[2+i]-8*y[3+i])/(120*h^3)
deriv_3_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(8*y[-5+i]-71*y[-4+i]+288*y[-3+i]-715*y[-2+i]+1000*y[-1+i]-693*y[i]+176*y[1+i]+7*y[2+i])/(120*h^3)
deriv_3_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(-7*y[-6+i]+64*y[-5+i]-267*y[-4+i]+680*y[-3+i]-1205*y[-2+i]+1392*y[-1+i]-889*y[i]+232*y[1+i])/(120*h^3)
deriv_3_node_8(y::AbstractVector{Float64},h::Float64,i::Int64)=(-232*y[-7+i]+1849*y[-6+i]-6432*y[-5+i]+12725*y[-4+i]-15560*y[-3+i]+11787*y[-2+i]-5104*y[-1+i]+967*y[i])/(120*h^3)

# depending on the node, compute the third derivative
compute_third_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_3_node_1(y,h,i) : node==2 ? deriv_3_node_2(y,h,i) : node==3 ? deriv_3_node_3(y,h,i) : node==4 ? deriv_3_node_4(y,h,i) : node==5 ? deriv_3_node_5(y,h,i) : node==6 ? deriv_3_node_6(y,h,i) : node==7 ? deriv_3_node_7(y,h,i) : node==8 ? deriv_3_node_8(y,h,i) : throw(DomainError(node, "node must be ≤ 8"))

# compute third derivative of points in at index compute_at
function compute_third_derivative(compute_at::Int64, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    return compute_third_derivative(y, h, compute_at, compute_node(compute_at, nPoints, 8))
end

# compute third derivative of points in array y
function compute_third_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_third_derivative(y, h, i, compute_node(i, nPoints, 8))
    end
end

# fourth derivative
deriv_4_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(3207*y[i]-21056*y[1+i]+61156*y[2+i]-102912*y[3+i]+109930*y[4+i]-76352*y[5+i]+33636*y[6+i]-8576*y[7+i]+967*y[8+i])/(240*h^4)
deriv_4_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(967*y[-1+i]-5496*y[i]+13756*y[1+i]-20072*y[2+i]+18930*y[3+i]-11912*y[4+i]+4876*y[5+i]-1176*y[6+i]+127*y[7+i])/(240*h^4)
deriv_4_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(127*y[-2+i]-176*y[-1+i]-924*y[i]+3088*y[1+i]-4070*y[2+i]+2928*y[3+i]-1244*y[4+i]+304*y[5+i]-33*y[6+i])/(240*h^4)
deriv_4_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(-33*y[-3+i]+424*y[-2+i]-1364*y[-1+i]+1848*y[i]-1070*y[1+i]+88*y[2+i]+156*y[3+i]-56*y[4+i]+7*y[5+i])/(240*h^4)
deriv_4_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(7*y[-4+i]-96*y[-3+i]+676*y[-2+i]-1952*y[-1+i]+2730*y[i]-1952*y[1+i]+676*y[2+i]-96*y[3+i]+7*y[4+i])/(240*h^4)
deriv_4_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(7*y[-5+i]-56*y[-4+i]+156*y[-3+i]+88*y[-2+i]-1070*y[-1+i]+1848*y[i]-1364*y[1+i]+424*y[2+i]-33*y[3+i])/(240*h^4)
deriv_4_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(-33*y[-6+i]+304*y[-5+i]-1244*y[-4+i]+2928*y[-3+i]-4070*y[-2+i]+3088*y[-1+i]-924*y[i]-176*y[1+i]+127*y[2+i])/(240*h^4)
deriv_4_node_8(y::AbstractVector{Float64},h::Float64,i::Int64)=(127*y[-7+i]-1176*y[-6+i]+4876*y[-5+i]-11912*y[-4+i]+18930*y[-3+i]-20072*y[-2+i]+13756*y[-1+i]-5496*y[i]+967*y[1+i])/(240*h^4)
deriv_4_node_9(y::AbstractVector{Float64},h::Float64,i::Int64)=(967*y[-8+i]-8576*y[-7+i]+33636*y[-6+i]-76352*y[-5+i]+109930*y[-4+i]-102912*y[-3+i]+61156*y[-2+i]-21056*y[-1+i]+3207*y[i])/(240*h^4)

# depending on the node, compute the fourth derivative
compute_fourth_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_4_node_1(y,h,i) : node==2 ? deriv_4_node_2(y,h,i) : node==3 ? deriv_4_node_3(y,h,i) : node==4 ? deriv_4_node_4(y,h,i) : node==5 ? deriv_4_node_5(y,h,i) : node==6 ? deriv_4_node_6(y,h,i) : node==7 ? deriv_4_node_7(y,h,i) : node==8 ? deriv_4_node_8(y,h,i) : node==9 ? deriv_4_node_9(y,h,i) : throw(DomainError(node, "node must be ≤ 9"))

# compute fourth derivative of points in at index compute_at
function compute_fourth_derivative(compute_at::Int64, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    return compute_fourth_derivative(y, h, compute_at, compute_node(compute_at, nPoints, 9))
end

# compute fourth derivative of points in array y
function compute_fourth_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_fourth_derivative(y, h, i, compute_node(i, nPoints, 9))
    end
end

# fifth derivative
deriv_5_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(-3013*y[i]+23421*y[1+i]-81444*y[2+i]+166476*y[3+i]-220614*y[4+i]+196638*y[5+i]-117876*y[6+i]+45804*y[7+i]-10461*y[8+i]+1069*y[9+i])/(144*h^5)
deriv_5_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(-1069*y[-1+i]+7677*y[i]-24684*y[1+i]+46836*y[2+i]-58014*y[3+i]+48774*y[4+i]-27852*y[5+i]+10404*y[6+i]-2301*y[7+i]+229*y[8+i])/(144*h^5)
deriv_5_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(-229*y[-2+i]+1221*y[-1+i]-2628*y[i]+2796*y[1+i]-1254*y[2+i]-306*y[3+i]+684*y[4+i]-372*y[5+i]+99*y[6+i]-11*y[7+i])/(144*h^5)
deriv_5_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(11*y[-3+i]-339*y[-2+i]+1716*y[-1+i]-3948*y[i]+5106*y[1+i]-4026*y[2+i]+2004*y[3+i]-636*y[4+i]+123*y[5+i]-11*y[6+i])/(144*h^5)
deriv_5_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(11*y[-4+i]-99*y[-3+i]+156*y[-2+i]+396*y[-1+i]-1638*y[i]+2334*y[1+i]-1716*y[2+i]+684*y[3+i]-141*y[4+i]+13*y[5+i])/(144*h^5)
deriv_5_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(-13*y[-5+i]+141*y[-4+i]-684*y[-3+i]+1716*y[-2+i]-2334*y[-1+i]+1638*y[i]-396*y[1+i]-156*y[2+i]+99*y[3+i]-11*y[4+i])/(144*h^5)
deriv_5_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(11*y[-6+i]-123*y[-5+i]+636*y[-4+i]-2004*y[-3+i]+4026*y[-2+i]-5106*y[-1+i]+3948*y[i]-1716*y[1+i]+339*y[2+i]-11*y[3+i])/(144*h^5)
deriv_5_node_8(y::AbstractVector{Float64},h::Float64,i::Int64)=(11*y[-7+i]-99*y[-6+i]+372*y[-5+i]-684*y[-4+i]+306*y[-3+i]+1254*y[-2+i]-2796*y[-1+i]+2628*y[i]-1221*y[1+i]+229*y[2+i])/(144*h^5)
deriv_5_node_9(y::AbstractVector{Float64},h::Float64,i::Int64)=(-229*y[-8+i]+2301*y[-7+i]-10404*y[-6+i]+27852*y[-5+i]-48774*y[-4+i]+58014*y[-3+i]-46836*y[-2+i]+24684*y[-1+i]-7677*y[i]+1069*y[1+i])/(144*h^5)
deriv_5_node_10(y::AbstractVector{Float64},h::Float64,i::Int64)=(1/(144*h^5))*(-1069*y[-9+i]+10461*y[-8+i]-45804*y[-7+i]+117876*y[-6+i]-196638*y[-5+i]+220614*y[-4+i]-166476*y[-3+i]+81444*y[-2+i]-23421*y[-1+i]+3013*y[i])

# depending on the node, compute the fifth derivative
compute_fifth_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_5_node_1(y,h,i) : node==2 ? deriv_5_node_2(y,h,i) : node==3 ? deriv_5_node_3(y,h,i) : node==4 ? deriv_5_node_4(y,h,i) : node==5 ? deriv_5_node_5(y,h,i) : node==6 ? deriv_5_node_6(y,h,i) : node==7 ? deriv_5_node_7(y,h,i) : node==8 ? deriv_5_node_8(y,h,i) : node==9 ? deriv_5_node_9(y,h,i) : node==10 ? deriv_5_node_10(y,h,i) : throw(DomainError(node, "node must be ≤ 10"))

# compute fifth derivative of points in at index compute_at
function compute_fifth_derivative(compute_at::Int64, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    return compute_fifth_derivative(y, h, compute_at, compute_node(compute_at, nPoints, 10))
end

# compute fifth derivative of points in array y
function compute_fifth_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_fifth_derivative(y, h, i, compute_node(i, nPoints, 10))
    end
end


# sixth derivative
deriv_6_node_1(y::AbstractVector{Float64},h::Float64,i::Int64)=(1/(240*h^6))*(7513*y[i]-67090*y[1+i]+270705*y[2+i]-650280*y[3+i]+1030290*y[4+i]-1125276*y[5+i]+858090*y[6+i]-451080*y[7+i]+156405*y[8+i]-32290*y[9+i]+3013*y[10+i])
deriv_6_node_2(y::AbstractVector{Float64},h::Float64,i::Int64)=(1/(240*h^6))*(3013*y[-1+i]-25630*y[i]+98625*y[1+i]-226440*y[2+i]+344010*y[3+i]-361716*y[4+i]+266730*y[5+i]-136200*y[6+i]+46065*y[7+i]-9310*y[8+i]+853*y[9+i])
deriv_6_node_3(y::AbstractVector{Float64},h::Float64,i::Int64)=(853*y[-2+i]-6370*y[-1+i]+21285*y[i]-42120*y[1+i]+55050*y[2+i]-50076*y[3+i]+32370*y[4+i]-14760*y[5+i]+4545*y[6+i]-850*y[7+i]+73*y[8+i])/(240*h^6)
deriv_6_node_4(y::AbstractVector{Float64},h::Float64,i::Int64)=(73*y[-3+i]+50*y[-2+i]-2355*y[-1+i]+9240*y[i]-18030*y[1+i]+21324*y[2+i]-16350*y[3+i]+8280*y[4+i]-2715*y[5+i]+530*y[6+i]-47*y[7+i])/(240*h^6)
deriv_6_node_5(y::AbstractVector{Float64},h::Float64,i::Int64)=(-47*y[-4+i]+590*y[-3+i]-2535*y[-2+i]+5400*y[-1+i]-6270*y[i]+3684*y[1+i]-390*y[2+i]-840*y[3+i]+525*y[4+i]-130*y[5+i]+13*y[6+i])/(240*h^6)
deriv_6_node_6(y::AbstractVector{Float64},h::Float64,i::Int64)=(13*y[-5+i]-190*y[-4+i]+1305*y[-3+i]-4680*y[-2+i]+9690*y[-1+i]-12276*y[i]+9690*y[1+i]-4680*y[2+i]+1305*y[3+i]-190*y[4+i]+13*y[5+i])/(240*h^6)
deriv_6_node_7(y::AbstractVector{Float64},h::Float64,i::Int64)=(13*y[-6+i]-130*y[-5+i]+525*y[-4+i]-840*y[-3+i]-390*y[-2+i]+3684*y[-1+i]-6270*y[i]+5400*y[1+i]-2535*y[2+i]+590*y[3+i]-47*y[4+i])/(240*h^6)
deriv_6_node_8(y::AbstractVector{Float64},h::Float64,i::Int64)=(-47*y[-7+i]+530*y[-6+i]-2715*y[-5+i]+8280*y[-4+i]-16350*y[-3+i]+21324*y[-2+i]-18030*y[-1+i]+9240*y[i]-2355*y[1+i]+50*y[2+i]+73*y[3+i])/(240*h^6)
deriv_6_node_9(y::AbstractVector{Float64},h::Float64,i::Int64)=(1/(240*h^6))*(73*y[-8+i]-850*y[-7+i]+4545*y[-6+i]-14760*y[-5+i]+32370*y[-4+i]-50076*y[-3+i]+55050*y[-2+i]-42120*y[-1+i]+21285*y[i]-6370*y[1+i]+853*y[2+i])
deriv_6_node_10(y::AbstractVector{Float64},h::Float64,i::Int64)=(1/(240*h^6))*(853*y[-9+i]-9310*y[-8+i]+46065*y[-7+i]-136200*y[-6+i]+266730*y[-5+i]-361716*y[-4+i]+344010*y[-3+i]-226440*y[-2+i]+98625*y[-1+i]-25630*y[i]+3013*y[1+i])
deriv_6_node_11(y::AbstractVector{Float64},h::Float64,i::Int64)=(1/(240*h^6))*(3013*y[-10+i]-32290*y[-9+i]+156405*y[-8+i]-451080*y[-7+i]+858090*y[-6+i]-1125276*y[-5+i]+1030290*y[-4+i]-650280*y[-3+i]+270705*y[-2+i]-67090*y[-1+i]+7513*y[i])

# depending on the node, compute the sixth derivative
compute_sixth_derivative(y::AbstractVector{Float64},h::Float64,i::Int64,node::Int64) = node==1 ? deriv_6_node_1(y,h,i) : node==2 ? deriv_6_node_2(y,h,i) : node==3 ? deriv_6_node_3(y,h,i) : node==4 ? deriv_6_node_4(y,h,i) : node==5 ? deriv_6_node_5(y,h,i) : node==6 ? deriv_6_node_6(y,h,i) : node==7 ? deriv_6_node_7(y,h,i) : node==8 ? deriv_6_node_8(y,h,i) : node==9 ? deriv_6_node_9(y,h,i) : node==10 ? deriv_6_node_10(y,h,i) : node==11 ? deriv_6_node_11(y,h,i) : throw(DomainError(node, "node must be ≤ 11"))

# compute sixth derivative of points in at index compute_at
function compute_sixth_derivative(compute_at::Int64, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    return compute_sixth_derivative(y, h, compute_at, compute_node(compute_at, nPoints, 11))
end

# compute sixth derivative of points in array y
function compute_sixth_derivative(deriv::AbstractVector{Float64}, y::AbstractVector{Float64}, h::Float64, nPoints::Int64)
    for i=1:nPoints
        deriv[i] = compute_sixth_derivative(y, h, i, compute_node(i, nPoints, 11))
    end
end

function compute_derivs(derivs::AbstractArray, ydata::AbstractVector{Float64},h::Float64, nPoints::Int64)
    compute_first_derivative(derivs[1], ydata,h, nPoints)
    compute_second_derivative(derivs[2], ydata,h, nPoints)
    # compute_third_derivative(derivs[3], ydata,h, nPoints)
    # compute_fourth_derivative(derivs[4], ydata,h, nPoints)
    # compute_fifth_derivative(derivs[5], ydata,h, nPoints)
    # compute_sixth_derivative(derivs[6], ydata,h, nPoints)
end

end