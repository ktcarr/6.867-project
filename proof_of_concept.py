#!/usr/bin/python3
import torch
import torch.autograd as tag

######################################
# CALCULATION 1: PARTIAL DERIVATIVES #
######################################
x = torch.tensor(3., requires_grad=True)
w = torch.tensor(5., requires_grad=True)

y = 2*x**3 + w*x**2 - 6*x
print(y)    # should equal 81

dydx = tag.grad(y, x, create_graph=True)[0]
print(dydx)     # should equal 78

d2ydx2 = tag.grad(dydx, x, create_graph=True)[0]
print(d2ydx2)   # should equal 46

# we can evaluate partial derivatives w.r.t. w one at a time
#f = y       # should equal 9
#f = dydx    # should equal 6
f = d2ydx2  # should equal 2

# i'm using .backward() here in place of tag.grad() to simulate
# how the neural network optimization function will expect
# gradients of the loss function to be evaluated.
#
# with .backward(), the gradients need to be explicitly zeroed
# after each call; my experimentation shows that multiple calls
# to tag.grad() followed by a single call to .backward()
# produces the correct derivatives without the explicit need to
# re-zero intermediate gradients.
f.backward()
print(w.grad)

#####################################
# CALCULATION 2: SUM OF DERIVATIVES #
#####################################
# reset
x = torch.tensor(3., requires_grad=True)
w = torch.tensor(5., requires_grad=True)

# recalculate gradients
y = 2*x**3 + w*x**2 - 6*x
dydx = tag.grad(y, x, create_graph=True)[0]
d2ydx2 = tag.grad(dydx, x, create_graph=True)[0]

# calculate the gradient w.r.t. w of the sum of all three
f = y + dydx + d2ydx2
f.backward()
print(w.grad)   # should equal 9 + 6 + 2 = 17
