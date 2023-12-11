import torch


######################
# A CDE model is defined as
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(
            hidden_channels,
            128,
        )  # torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(
            128,
            input_channels * hidden_channels,
        )  # torch.nn.Linear(hidden_channels, input_channels * hidden_channels)

        self.W = torch.nn.Parameter(torch.Tensor(input_channels))
        self.W.data.fill_(1)

        self.ode_case = False

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)

        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        # z = torch.matmul(z,torch.diag(self.W))

        if self.ode_case:
            z = torch.mean(z, dim=2)

        return z
