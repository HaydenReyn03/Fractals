import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
Y, X = np.mgrid[-1:1:0.05,-1:1:0.05]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y) #important!
ns = torch.zeros_like(z)

# cos(60)
x60 = np.cos(np.pi/3)
y60 = np.sin(np.pi/3)
z60 = torch.complex(torch.tensor(x60), torch.tensor(y60))

length = torch.tensor(0.0)
seg_length = torch.tensor(0.0)
num_lines = torch.tensor(0.0)

# Transfer to the GPU device
z = z.to(device)
z60 = z60.to(device)

l = length.to(device)
seg_l = seg_length.to(device)
num = num_lines.to(device)

## Setup points register and initial point
zero = torch.tensor(0.0)
one = torch.tensor(1.0)

points = torch.tensor([], dtype=torch.complex64, device=device)

#Koch Curve
for i in range(3):
    l = torch.pow(torch.tensor(4/3), i)
    seg_l = torch.pow(torch.tensor(1/3), i)
    num = torch.pow(torch.tensor(4), i)

    p = torch.tensor([], dtype=torch.complex64, device=device)
    if i == 0:
        p = torch.cat([points, torch.complex(zero,zero).unsqueeze(0)])
    
    for j in range(len(points) - 1):
        angle = torch.angle(points[j+1] - points[j])
        p = torch.cat([p, points[j].unsqueeze(0)])
        print("Iter:", i, "Number of points:", len(p))
        p = torch.cat([p, p[j].unsqueeze(0) + torch.complex(seg_l * torch.cos(angle), seg_l * torch.sin(angle))])
        print("Iter:", i, "Number of points:", len(p))
        p = torch.cat([p, p[4*j+1].unsqueeze(0) + torch.complex(seg_l * torch.cos(angle+(np.pi/3)), seg_l * torch.sin(angle+(np.pi/3)))])
        print("Iter:", i, "Number of points:", len(p))
        p = torch.cat([p, p[4*j+1].unsqueeze(0) + torch.complex(seg_l * torch.cos(angle), seg_l * torch.sin(angle))])
        print("Iter:", i, "Number of points:", len(p))
    p = torch.cat([p, torch.complex(zero,one).unsqueeze(0)])
    print("Iteration:", i, "Number of points:", len(p))
    points = p.clone()

        
#plot
plt.plot(points.cpu().numpy().real, points.cpu().numpy().imag, color='blue', linewidth=1)
plt.tight_layout(pad=0)
plt.show()