import torch
import torch.nn as nn

def example1():
    s=torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)
    s=torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    s=torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    s=torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x=torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)
class ToyModel(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.fc1(x))
        print(f"fc1 output dtype: {x.dtype}")
        x = self.ln(x)
        print(f"LayerNorm output dtype: {x.dtype}")
        x = self.fc2(x)
        return x
def example2():
    model = ToyModel(20, 5).cuda()
    x = torch.randn(4, 20).cuda()
    y = torch.randint(0, 5, (4,)).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(x)
        print(f"Final output dtype: {logits.dtype}")
        loss = nn.CrossEntropyLoss()(logits, y)
        print(f"Loss dtype: {loss.dtype}")
        loss.backward()
        print(f"Gradients dtype: {model.fc1.weight.grad.dtype}")
if __name__ == "__main__":
    # example1()
    example2()