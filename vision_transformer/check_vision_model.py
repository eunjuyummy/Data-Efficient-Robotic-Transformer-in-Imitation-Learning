import timm
model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
print(model)