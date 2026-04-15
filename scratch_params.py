from src.model.siamese_snn import SiameseSNN

model = SiameseSNN(
    in_channels=4,
    encoder_channels=[32, 64, 128, 256],
    num_steps=8
)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
