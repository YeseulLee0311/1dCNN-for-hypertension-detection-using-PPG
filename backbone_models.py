import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(128)
            )),
            ResBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2, downsample=nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(256)
            )),
            ResBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2, downsample=nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(512)
            )),
            ResBlock(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.conv7 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv8 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(512)
        self.conv9 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(512)
        self.conv10 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.conv11 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm1d(512)
        self.conv12 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm1d(512)
        self.conv13 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm1d(512)
        self.pool5 = nn.MaxPool1d(kernel_size=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        '''self.fc1 = nn.Linear(512, 4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, num_classes)'''
        
        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.functional.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.functional.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.functional.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.functional.relu(x)
        x = self.pool4(x)
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = nn.functional.relu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = nn.functional.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = nn.functional.relu(x)
        x = self.pool5(x)
        
        x = self.avgpool(x)
        
        x = self.fc(x)
        
        return x