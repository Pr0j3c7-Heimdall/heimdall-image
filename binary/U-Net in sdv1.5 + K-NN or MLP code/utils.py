import torch
from diffusers import StableDiffusionPipeline
from torchvision import transforms

def get_transform():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

class FeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"Loading SDM v1.5...")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        
        self.pipe.set_progress_bar_config(disable=True)
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        
        self.features = {}
        self._register_hooks()

    def _hook_fn(self, module, input, output):
        self.features['decoder_16'] = torch.mean(output, dim=(2, 3))

    def _register_hooks(self):
        layer_to_hook = self.unet.up_blocks[0]
        layer_to_hook.register_forward_hook(self._hook_fn)

    @torch.no_grad()
    def extract_features_batch(self, images_tensor):
        pipeline_device = self.pipe.device
        images_tensor = images_tensor.to(pipeline_device)

        if self.pipe.dtype == torch.float16:
            images_tensor = images_tensor.half()

        # [수정] .sample() 대신 .mode() 적용하여 결정론적 특징 추출
        latents = self.vae.encode(images_tensor).latent_dist.mode()
        latents = latents * self.vae.config.scaling_factor

        batch_size = images_tensor.shape[0]
        timesteps = torch.zeros((batch_size,), device=pipeline_device, dtype=torch.long)
        
        dummy_text_input = self.pipe.tokenizer(
            [""] * batch_size, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.pipe.tokenizer.model_max_length, 
            truncation=True
        )
        encoder_hidden_states = self.pipe.text_encoder(dummy_text_input.input_ids.to(pipeline_device))[0]
        
        if self.pipe.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.half()

        self.unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states)
        
        # [수정] numpy 변환 없이 순수 PyTorch Tensor 반환
        return self.features['decoder_16'].float().cpu()