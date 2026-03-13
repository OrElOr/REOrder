import torch
import torch.nn as nn
import timm

class SPEAR_Swin(nn.Module):
    """
    Experimental wrapper for Swin Transformer that applies Naive Global Permutation (SPEAR).
    It globally scrambles the patches immediately after patch embedding, before any Window Partitioning or Merge.
    """
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=10, pretrained=False):
        super().__init__()
        
        # 1. Load the base Swin model from timm
        # swin_tiny_patch4_window7_224 uses 224x224 images and 4x4 patches.
        # This results in (224//4) * (224//4) = 56 * 56 = 3136 initial patches.
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # 2. Extract number of patches from the patch embedding layer
        self.num_patches = self.base_model.patch_embed.num_patches
        
        # 3. Generate the global secret permutation pi (SPEAR mechanism)
        # Randomly shuffles the integers from 0 to num_patches - 1
        secret_pi = torch.randperm(self.num_patches)
        
        # Register as a buffer so it saves with the model and automatically moves to GPU/CPU
        self.register_buffer('secret_pi', secret_pi)

    def forward_features(self, x):
        """
        Overrides the standard forward_features to inject the permutation immediately
        after the image is broken into patches, but before positional dropout and the transformer layers.
        """
        # [Step 1] Patch Embedding: x shape becomes (B, L, C)
        # For Swin-Tiny: (B, 3136, 96)
        x = self.base_model.patch_embed(x)
        
        # If absolute positional embedding is used, add it
        if self.base_model.absolute_pos_embed is not None:
            x = x + self.base_model.absolute_pos_embed

        # ==========================================================
        # [Step 2] SPEAR DEFENSE: NAIVE GLOBAL PERMUTATION
        # ==========================================================
        B, L, C = x.shape
        
        # Ensure the secret_pi matches the sequence length (in case of different image sizes)
        assert L == len(self.secret_pi), f"Expected sequence length {len(self.secret_pi)}, got {L}"

        # Expand secret_pi to match shape (B, L, C)
        # 1) secret_pi is (L,) -> (1, L, 1)
        # 2) expand to (B, L, C)
        pi_expanded = self.secret_pi.unsqueeze(0).unsqueeze(-1).expand(B, L, C)
        
        # Scramble the sequence dimension using the secret permutation
        x = torch.gather(x, dim=1, index=pi_expanded)
        # ==========================================================

        # [Step 3] Proceed into the rest of the Swin architecture
        x = self.base_model.pos_drop(x)
        x = self.base_model.layers(x)
        x = self.base_model.norm(x)
        
        return x

    def forward(self, x):
        # Pass through the modified feature extractor
        x = self.forward_features(x)
        
        # Pass through the classification head
        x = self.base_model.head(x)
        return x
