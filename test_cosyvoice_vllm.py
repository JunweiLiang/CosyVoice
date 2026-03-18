# coding=utf-8
# test cosyvoice

import sys
import time
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import sounddevice as sd

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
    output_device_idx = sd.default.device[0]
    print("Use this speaker: %s" % devices[output_device_idx]["name"])


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
    print("model sample rate from config:" % cosyvoice.sample_rate)
    print("----load model done -----")

    def generate_voice(output):
        for i, j in enumerate(output):
            print(i)
            #torchaudio.save('zero_shot_{}_laoban.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
            # 只支持macOS
            #torchaudio.io.play_audio(j['tts_speech'], cosyvoice.sample_rate, device=0)

            #https://python-sounddevice.readthedocs.io/en/0.5.1/usage.html#playback

            #print(type(j["tts_speech"]), j["tts_speech"].shape)
            # <class 'torch.Tensor'> torch.Size([1, 222720])
            sd.play(j['tts_speech'].cpu().detach().numpy().T, cosyvoice.sample_rate)
            print("took %.3f seconds" % (time.perf_counter() - start_time))
            sd.wait() # wait till the playback is complete
            # sd.stop() # will make it stop mid play


    # 微信语音，然后用苹果电脑quicktime录声音，然后转
    # ffmpeg -i /Users/junweiliang/Downloads/zero_shot_prompt_laopo.m4a -ss 00:00:03 -to 00:00:06 -acodec pcm_s16le -ac 1 -ar 16000 zero_shot_prompt_laopo.wav
    # if you want to slow down the audio: -filter:a "atempo=0.8"


    trys = 3 # 4 次
    for i in range(trys):
        start_time = time.perf_counter()
        output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
            prompt_speech_text, wav_file, stream=False) # stream=True 下面才会有多个segment，效果很差，会卡

        generate_voice(output)
        start_time = time.perf_counter()
        output = cosyvoice.inference_zero_shot_fast('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
            stream=False) # stream=True 下面才会有多个segment，效果很差，会卡

        generate_voice(output)
        print("--------- try again---------")

