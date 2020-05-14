'''
touch.Tensor
    requires_grad = True , 연산을 기록함
        .backward(): gradient 자동 계산, 스칼라는 필요x
        .grad()에 누적
        .detach(): track 중단
        with torch.no_grad: 메모리사용&기록 x, evaluate할 때 유용
'''

import torch

#Tensor 연산 기록
x = torch.ones(2,2, requires_grad=True)
print(x)

y = x + 2
print(y)

#연산의 결과로 생성된 y는 grad_fn을 가짐
print(y.grad_fn) # <AddBackward0 object at 0x000001F9C9F4A208>

z = y * y * 3
out = z.mean()
print(z, out) # tensor([[27., 27.], [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True) # 바꾸기 True로
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn) # <SumBackward0 object at 0x000002AB0540A9E8>

out.backward() # out.backward(torch.tensor(1.)): out은 스칼라값이므로

print(x.grad) # d(out)/dx

#vector-Jacobian Matrix multiplex example
x = torch.randn(3, requires_grad=True)
print(x)

y = x * 2
while y.data.norm() < 1000:
    print(y)
    print(y.data.norm())
    print(torch.sqrt(torch.sum(torch.pow(y, 2)))) # L2 distance
    y = y * 2

print(y)

# v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# y.backward(v)
#
# print(x.grad)

print(x.requires_grad) #True
print((x ** 2).requires_grad) #True

with torch.no_grad():
    print((x ** 2).requires_grad) #False

# print(x.requires_grad) #True
# y = x.detach()
# print(y.requires_grad) #False
# print(x.eq(y).all()) #tensor(True)

