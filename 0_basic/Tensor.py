#study : Tensor
#https://tutorials.pytorch.kr/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
# from __future__ import print_function #python2에서 3을 사용할 때
import torch
import numpy as np

#초기화되지 않은 5x3행렬 생성
x = torch.empty(5,3)
print(x) #행렬이 생성되면 그 시점에 할당된 메모리에 존재하던 값들이 초기값으로 나타납니다.

#무작위로 초기화된 행렬을 생성
x = torch.rand(5, 3)
print(x)

#dtype이 long이고 0으로 채워진 행렬을 생성
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#직접 생성
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* 메소드는 크기를 받습니다
print(x)

x = torch.randn_like(x, dtype=torch.float)    # dtype을 오버라이드(Override) 합니다! #기존 텐서를 쓸 때 사용함수
print(x)                                      # 결과는 동일한 크기를 갖습니다

print(x.size())

#연산
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

#덧셈: 결과 tensor를 인자로 제공
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

#y에 x 더하기
y.add_(x)
print(y)

print(x)
print(x[:, 1])

#크기 변경: tensor의 크기(size)나 모양(shape)을 변경
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1은 다른 차원들을 사용하여 유추합니다.
r = x.view(-1, 1)
print(x.size(), y.size(), z.size())
print(x,'\n',y,'\n',z, '\n',r)

#tensor에 하나의 값만 존재한다면, .item() 을 사용하면 숫자 값 얻기 가능
x = torch.randn(1)
print(x)
print(x.item())

#tensor to numpy
print(y.numpy())

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b.shape[0])
print(b)
print([b[i] for i in range(0,b.shape[0])])

#cuda tensor
# 이 코드는 CUDA가 사용 가능한 환경에서만 실행합니다.
# ``torch.device`` 를 사용하여 tensor를 GPU 안팎으로 이동해보겠습니다.
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # CUDA 장치 객체(device object)로
#     y = torch.ones_like(x, device=device)  # GPU 상에 직접적으로 tensor를 생성하거나
#     x = x.to(device)                       # ``.to("cuda")`` 를 사용하면 됩니다.
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # ``.to`` 는 dtype도 함께 변경합니다!