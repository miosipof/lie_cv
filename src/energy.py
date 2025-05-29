import torch
import torch.nn.functional as F
import torch.nn as nn

class FieldOptimizer2D:
    def __init__(self):
        pass

    def apply_lie_flow(self, S, phi_fields, eps=1.0):
        # Sobel-style horizontal and vertical derivative kernels
        kernel_dx = torch.tensor([[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).to(S.device) / 8.0
        
        kernel_dy = torch.tensor([[[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0).to(S.device) / 8.0


        phi0 = phi_fields[0]
        phi1 = phi_fields[1]
        phi2 = phi_fields[2]

        # Compute derivatives
        dS_dx = F.conv2d(S, kernel_dx, padding=1)
        dS_dy = F.conv2d(S, kernel_dy, padding=1)

        B, C, H, W = S.shape

        # Coordinate grids
        y_coords, x_coords = torch.meshgrid(torch.linspace(-1, 1, H, device=S.device),
                                            torch.linspace(-1, 1, W, device=S.device), indexing='ij')
        x_coords = x_coords.view(1, 1, H, W).expand(B, 1, H, W)
        y_coords = y_coords.view(1, 1, H, W).expand(B, 1, H, W)

        # Rotation generator: x dS/dy - y dS/dx
        rot_generator = x_coords * dS_dy - y_coords * dS_dx

        # Combine Lie terms
        lie_update = (
            phi0 * dS_dx +
            phi1 * dS_dy +
            phi2 * rot_generator
        )

        S_prime = S + eps * lie_update
        return S_prime


    def compute_energy(self, S, coupling_constant=1.0, alpha=0.1, beta=0.1, v=0.1,
                                num_steps=50, lr=0.05, coarse_factor=8):
        B, C, H, W = S.shape
        Hc, Wc = H // coarse_factor, W // coarse_factor

        # Initialize sparse phi fields (on coarse grid)
        phi_sparse = [
            (0.1 * v + 1e-4 * torch.randn((B, 1, Hc, Wc), device=S.device)).requires_grad_()
            for _ in range(3)
        ]
        optimizer = torch.optim.Adam(phi_sparse, lr=lr)

        # Sobel kernels for smoothness (on coarse grid)
        kernel_dx = torch.tensor([[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).to(S.device) / 8.0
        kernel_dy = torch.tensor([[[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0).to(S.device) / 8.0

        void_mask = (S < 0.05).float()  # shape: (B, 1, H, W), mask where S is close to 0


        for step in range(num_steps):
            # Interpolate phi to image resolution
            phi_dense = [F.interpolate(f, size=(H, W), mode='bilinear', align_corners=True) for f in phi_sparse]

            # Apply Lie flow
            S_prime = self.apply_lie_flow(S, phi_dense, eps=1.0)

            # Coupling loss
            coupling = F.mse_loss(S_prime, S)

            # Smoothness on coarse phi
            kinetic = 0.0
            for f in phi_sparse:
                dphi_dx = F.conv2d(f, kernel_dx, padding=1)
                dphi_dy = F.conv2d(f, kernel_dy, padding=1)
                kinetic += alpha * (
                    F.mse_loss(dphi_dx, torch.zeros_like(dphi_dx)) +
                    F.mse_loss(dphi_dy, torch.zeros_like(dphi_dy))
                )

            phi_norm = sum([torch.norm(f) for f in phi_sparse])
            ssb = beta * torch.abs(phi_norm**2 - v**2)

            # if step % 10 == 0:
            #     print(f"[Step {step}] Coupling: {coupling.item():.4e}, Kinetic: {kinetic.item():.4e}, SSB: {ssb.item():.4e}")


            gamma = 0.07  # weight for void penalty
            void_penalty = 0.0
            
            for phi_i in phi_dense:
                void_penalty += gamma * torch.sum(void_mask * phi_i**2)
            
            # Total loss
            E = coupling_constant * coupling + kinetic + ssb + gamma*void_penalty

            optimizer.zero_grad()
            E.backward()
            optimizer.step()

        return S_prime, [f.detach() for f in phi_sparse], [F.interpolate(f, size=(H, W), mode='bilinear', align_corners=True).detach() for f in phi_sparse]




class FieldOptimizerShifts(torch.nn.Module):
    def __init__(self, B, C, H, W, coupling_constant=1.0, alpha=0.1, beta=0.1, v=0.1, lr=0.05, coarse_factor=1, num_steps=50, lie_dim=2):
        super().__init__()

        self.coupling_constant = coupling_constant
        self.alpha = alpha
        self.beta = beta
        self.v = v

        self.coarse_factor = coarse_factor
        self.num_steps = num_steps
        self.lr = lr
        self.lie_dim = lie_dim

        # Initialize sparse phi fields (on coarse grid)
        self.phi_sparse = torch.nn.ParameterList([
            torch.nn.Parameter(0.1 * v + 1e-4 * torch.randn(B, C, H//self.coarse_factor, W//self.coarse_factor))
            for _ in range(self.lie_dim)
        ])

        self.upsample = nn.Upsample(
            scale_factor=self.coarse_factor,
            mode='bilinear',
            align_corners=True
        )

        self.optimizer = torch.optim.Adam(self.phi_sparse, lr=self.lr)


    def apply_lie_flow(self, S, phi_fields, eps=1.0):
        """
        Apply one Euler‐step of a “Lie flow” to a multichannel field S.

        Args:
        S          Tensor [B, C, H, W]         — feature maps
        phi_fields List of tensors             - [phi_x, phi_y], each [B, C, H, W]
        eps        float                       — step size

        Returns:
        S_prime    Tensor [B, C, H, W]
        """
        # unpack and sanity‐check
        assert len(phi_fields) == 2,       "Expect exactly two phi fields (x and y)"
        phi_x, phi_y = phi_fields
        assert phi_x.shape == S.shape,     f"phi_x {phi_x.shape} != S {S.shape}"
        assert phi_y.shape == S.shape,     f"phi_y {phi_y.shape} != S {S.shape}"

        # compute forward differences
        # dS/dx  (difference along width dim=3)
        dx = torch.zeros_like(S)
        dx[..., :-1] = S[..., 1:] - S[..., :-1]
        # dS/dy  (difference along height dim=2)
        dy = torch.zeros_like(S)
        dy[..., :-1, :] = S[..., 1:, :] - S[..., :-1, :]

        # Lie‐update: phi_x * dS/dx + phi_y * dS/dy
        lie_update = phi_x * dx + phi_y * dy

        # Euler step
        S_prime = S + eps * lie_update
        return S_prime

    

    def compute_energy(self, S):
        # Latent representation: H = W = 28 = 224/8 = img_size/patch_size
        # C = latent dimension        
        B, C, H, W = S.shape
        Hc, Wc = H // self.coarse_factor, W // self.coarse_factor

        # Sobel kernels for smoothness (on coarse grid)
        kernel_dx = torch.tensor([[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).to(S.device) / 8.0
        kernel_dy = torch.tensor([[[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0).to(S.device) / 8.0
                
        # Replicate the single‐channel kernel C times → [C,1,3,3]
        kernel_dx = kernel_dx.repeat(C, 1, 1, 1)
        kernel_dy = kernel_dy.repeat(C, 1, 1, 1)

        void_mask = (S < 0.05).float()  # shape: (B, 1, H, W), mask where S is close to 0

        for step in range(self.num_steps):
            # Interpolate phi to image resolution
            # phi_dense = [F.interpolate(f, size=(H, W), mode='bilinear', align_corners=True) for f in phi_sparse]         
            phi_dense = [self.upsample(f) for f in self.phi_sparse]

            # Apply Lie flow
            S_prime = self.apply_lie_flow(S, phi_dense, eps=1.0) # (B, C, H, W)

            # Coupling loss
            coupling = F.mse_loss(S_prime, S)

            # Smoothness on coarse phi
            kinetic = 0.0
            for f in self.phi_sparse:
                # Depthwise conv: groups=C
                dphi_dx = F.conv2d(f, kernel_dx, padding=1, groups=C)  # → [B, C, H, W]
                dphi_dy = F.conv2d(f, kernel_dy, padding=1, groups=C)  # → [B, C, H, W]
                kinetic += self.alpha * (
                    F.mse_loss(dphi_dx, torch.zeros_like(dphi_dx)) +
                    F.mse_loss(dphi_dy, torch.zeros_like(dphi_dy))
                )

            phi_norm = sum([torch.norm(f) for f in self.phi_sparse])
            ssb = self.beta * torch.abs(phi_norm**2 - self.v**2)

            gamma = 0.07  # weight for void penalty
            void_penalty = 0.0
            
            for phi_i in phi_dense:
                void_penalty += gamma * torch.sum(void_mask * phi_i**2)
            
            # Total loss
            E = self.coupling_constant * coupling + kinetic + ssb + gamma*void_penalty

            self.optimizer.zero_grad()
            E.backward()
            self.optimizer.step()

            # if step % 10 == 0:
            #     print(f"[Step {step}] E: {E.item():.4e}, Coupling: {coupling.item():.4e}, Kinetic: {kinetic.item():.4e}, SSB: {ssb.item():.4e}")            

        return S_prime, [f.detach() for f in self.phi_sparse], [f.detach() for f in phi_dense]
