from argparse import Namespace

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from model.multiview_rf.mv_sd3_architecture import MultiViewSD3Transformer


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


# Load a sd multi-view rf model
def create_sd_multiview_rf_model(num_input_output_channel=38):
    args = Namespace()
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    args.revision = None
    args.variant = None

    # rf model
    rf_transformer = MultiViewSD3Transformer.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="transformer",
        ignore_mismatched_sizes=True,
        strict=False,
        low_cpu_mem_usage=False,
    )
    rf_transformer.adjust_output_input_channel_size(num_input_output_channel)

    # tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        low_cpu_mem_usage=True,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        low_cpu_mem_usage=True,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
        low_cpu_mem_usage=True,
    )

    # Text encoders
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    def load_text_encoders(class_one, class_two, class_three):
        text_encoder_one = class_one.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
        text_encoder_two = class_two.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=args.revision,
            variant=args.variant,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
        text_encoder_three = class_three.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            revision=args.revision,
            variant=args.variant,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
        return text_encoder_one, text_encoder_two, text_encoder_three

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    return rf_transformer, tokenizers, text_encoders
