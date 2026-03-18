# coding=utf-8
# test cosyvoice

import sys
import time
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import sounddevice as sd
import torchaudio.functional as F
import torchaudio
import torch

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel

if __name__ == "__main__":
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No speaker devices found")
        sys.exit(0)
    print(devices)
    input_device_idx, output_device_idx = sd.default.device
    print("default is using this speaker: %s" % devices[output_device_idx]["name"])


    # zero_shot usage
    print("----Loading wav -----")
    wav_file = './test_audio/laopo2.wav'
    prompt_speech_16k = load_wav(wav_file, 16000)

    # cosyvoice 3 比 2多了这个prompt: You are a helpful assistant.<|endofprompt|> 必须要加
    prompt_speech_text = "You are a helpful assistant.<|endofprompt|>现在我们有很多突出的矛盾，比如说人岗不匹配，比如说这个整个学科设置不合理，那么就整个会导致我们培养出来的学生的能力，和真正的市场需求，他是脱节的。那么这个问题为什么会产生呢，一方面是因为现在整个科技的发展在加速，导致整个用工市场，对能力的需求的结构，也是在快速地变化。"
    print("----Loaded wav -----")

    """
    cosyvoice = CosyVoice2('./pretrained_models/CosyVoice2-0.5B_vllm/CosyVoice2-0.5B',
        load_jit=False, load_trt=False, fp16=True, load_vllm=True,
        prompt_text=prompt_speech_text, prompt_speech_16k=prompt_speech_16k, gpu_memory_utilization=0.5, quantization="fp8")#quantization=None)
    """
    print("----Loading model -----")
    cosyvoice = AutoModel(
        model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_vllm=True,
        prompt_text=prompt_speech_text, prompt_speech_16k=prompt_speech_16k)
    print("model sample rate from config: %s" % cosyvoice.sample_rate) # 24000
    print("----load model done -----")

    def generate_voice(output):
        for i, j in enumerate(output):
            print(f"Processing segment {i}")

            # 1. Grab tensor, force it to standard float32
            audio_tensor = j['tts_speech'].cpu().detach().to(torch.float32)

            # 2. Normalize the audio to the [-1.0, 1.0] range to prevent clipping/silence
            max_val = max(abs(audio_tensor.max()), abs(audio_tensor.min()))
            if max_val > 0:
                audio_tensor = audio_tensor / max_val

            # 3. Resample to 48000Hz (from our previous fix)
            target_sr = 48000
            resampled_audio = F.resample(
                audio_tensor,
                orig_freq=cosyvoice.sample_rate,
                new_freq=target_sr
            )

            # 4. Save to a WAV file (Guaranteed to work if you are on a remote server!)
            #file_name = f'zero_shot_output_{i}.wav'
            #torchaudio.save(file_name, resampled_audio, target_sr)
            #print(f"Saved audio to {file_name}")

            # 5. Try playing it locally (Will only be heard if running locally)
            audio_np = resampled_audio.numpy().T
            sd.play(audio_np, target_sr, device=8)

            print("took %.3f seconds" % (time.perf_counter() - start_time))
            sd.wait()


    # 微信语音，然后用苹果电脑quicktime录声音，然后转
    # ffmpeg -i /Users/junweiliang/Downloads/zero_shot_prompt_laopo.m4a -ss 00:00:03 -to 00:00:06 -acodec pcm_s16le -ac 1 -ar 16000 zero_shot_prompt_laopo.wav
    # if you want to slow down the audio: -filter:a "atempo=0.8"


    trys = 1 # 4 次
    for i in range(trys):
        start_time = time.perf_counter()
        output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
            prompt_speech_text, wav_file, stream=False) # stream=True 下面才会有多个segment，效果很差，会卡

        generate_voice(output)
        sys.exit()
        start_time = time.perf_counter()
        output = cosyvoice.inference_zero_shot_fast('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
            stream=False) # stream=True 下面才会有多个segment，效果很差，会卡

        generate_voice(output)
        print("--------- try again---------")

