"""
Post-production Finetuning Smoke Test Suite

Runs one single-sample finetune update for all modalities:
1. Drawing
2. Tremor
3. Audio
4. Questionnaire

All models are loaded from weights/ and saved back into weights/.
"""

from pathlib import Path

import joblib
import torch

from utils.helper_functions import load_user_data, print_section_header


# =========================
# Paths
# =========================
ROOT = Path(__file__).parent
WEIGHTS_DIR = ROOT / "weights"
EXAMPLES_DIR = ROOT / "examples"
CONFIG_PATH = ROOT / "finetune" / "config.yaml"


def _print_finetune_result(label: str, result: dict, saved_path: Path) -> None:
	prob = result.get("probability", 0.0)
	loss = result.get("loss", 0.0)
	pred = result.get("predicted_label", "?")
	target = result.get("target_label", "?")
	print(f"{label:20s} | loss={loss:.4f} | prob={prob:.4f} | pred={pred} | target={target}")
	print(f"{'':20s} | saved -> {saved_path.as_posix()}")


def run_drawing_finetune() -> None:
	print_section_header("1. DRAWING FINETUNE")

	from models.mobilenetV3 import MobileNetV3LargeBinary
	from predict_from_drawing import preprocess_drawing, get_transforms
	from finetune.finetune_drawing import finetune as finetune_drawing

	model = MobileNetV3LargeBinary(pretrained=False)
	image_path = EXAMPLES_DIR / "Healthy1.png"
	image_np = preprocess_drawing(str(image_path))
	image_tensor = get_transforms()(image=image_np)["image"]
	label = torch.tensor([0.0], dtype=torch.float32)

	load_ckpt = WEIGHTS_DIR / "Spiral_Drawing_Model.pth"
	save_ckpt = WEIGHTS_DIR / "finetuned/Spiral_Drawing_Model_finetuned.pth"

	result = finetune_drawing(
		model=model,
		image=image_tensor,
		label=label,
		load_pretrained=str(load_ckpt),
		save_path=str(save_ckpt),
		config_path=str(CONFIG_PATH),
		steps=1,
	)
	_print_finetune_result("Drawing", result, save_ckpt)


def run_tremor_finetune() -> None:
	print_section_header("2. TREMOR FINETUNE")

	from models.tremorNet import TremorClassifier
	from predict_from_tremor import extract_features_from_txt
	from finetune.finetune_tremor import finetune as finetune_tremor

	left = EXAMPLES_DIR / "tremor" / "healthy" / "001_CrossArms_LeftWrist.txt"
	right = EXAMPLES_DIR / "tremor" / "healthy" / "001_CrossArms_RightWrist.txt"

	features_array = extract_features_from_txt(str(left), str(right))
	scaler = joblib.load(WEIGHTS_DIR / "tremor_scaler.pkl")
	features_array = scaler.transform(features_array)

	features = torch.tensor(features_array[0], dtype=torch.float32)
	handedness = torch.tensor([1], dtype=torch.long)  # right-handed
	movement = torch.tensor([0], dtype=torch.long)    # CrossArms
	label = torch.tensor([0.0], dtype=torch.float32)

	model = TremorClassifier()
	load_ckpt = WEIGHTS_DIR / "Tremor_Model.pth"
	save_ckpt = WEIGHTS_DIR / "finetuned/Tremor_Model_finetuned.pth"

	result = finetune_tremor(
		model=model,
		features=features,
		handedness=handedness,
		movement=movement,
		label=label,
		load_pretrained=str(load_ckpt),
		save_path=str(save_ckpt),
		config_path=str(CONFIG_PATH),
		per_movement=False,
		steps=1,
	)
	_print_finetune_result("Tremor", result, save_ckpt)


def run_audio_finetune() -> None:
	print_section_header("3. AUDIO FINETUNE")

	from models.densenet169 import DenseNet1691D
	from predict_from_audio import extract_features_from_wav, preprocess_audio_input
	from finetune.finetune_audio import finetune as finetune_audio

	wav_path = EXAMPLES_DIR / "healthy_audio.wav"
	scaler = joblib.load(WEIGHTS_DIR / "audio_scaler.save")
	feature_vector = extract_features_from_wav(str(wav_path), gender="M")
	features = preprocess_audio_input(feature_vector, scaler).squeeze(0)
	label = torch.tensor([0.0], dtype=torch.float32)

	model = DenseNet1691D(pretrained=False)
	load_ckpt = WEIGHTS_DIR / "Audio_Tabular_Model.pth"
	save_ckpt = WEIGHTS_DIR / "finetuned/Audio_Tabular_Model_finetuned.pth"

	result = finetune_audio(
		model=model,
		features=features,
		label=label,
		load_pretrained=str(load_ckpt),
		save_path=str(save_ckpt),
		config_path=str(CONFIG_PATH),
		steps=1,
	)
	_print_finetune_result("Audio", result, save_ckpt)


def run_questionnaire_finetune() -> None:
	print_section_header("4. QUESTIONNAIRE FINETUNE")

	from models.densenet169 import DenseNet1691D
	from predict_from_questionnaire import preprocess_tabular_input
	from finetune.finetune_questionnaire import finetune as finetune_questionnaire

	user_data = load_user_data(str(EXAMPLES_DIR / "user_data.yaml"))
	questionnaire_input = [
		user_data["age"],
		user_data["height"],
		user_data["weight"],
		user_data["gender"],
		user_data["appearance_in_kinship"],
		user_data["appearance_in_first_grade_kinship"],
		user_data["questions"],
	]

	features = preprocess_tabular_input(questionnaire_input).squeeze(0)
	label = torch.tensor([0.0], dtype=torch.float32)

	model = DenseNet1691D(pretrained=False)
	load_ckpt = WEIGHTS_DIR / "Metadata_Model.pth"
	save_ckpt = WEIGHTS_DIR / "finetuned/Metadata_Model_finetuned.pth"

	result = finetune_questionnaire(
		model=model,
		features=features,
		label=label,
		load_pretrained=str(load_ckpt),
		save_path=str(save_ckpt),
		config_path=str(CONFIG_PATH),
		steps=1,
	)
	_print_finetune_result("Questionnaire", result, save_ckpt)


def run_tests(mode: str = "all") -> None:
	"""
	mode:
		- "drawing"
		- "tremor"
		- "audio"
		- "questionnaire"
		- "all"
	"""
	mode = mode.lower()

	if mode in ("drawing", "all"):
		run_drawing_finetune()

	if mode in ("tremor", "all"):
		run_tremor_finetune()

	if mode in ("audio", "all"):
		run_audio_finetune()

	if mode in ("questionnaire", "all"):
		run_questionnaire_finetune()


if __name__ == "__main__":
	run_tests(mode="all")
