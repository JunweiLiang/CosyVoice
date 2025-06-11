# coding=utf-8
# test cosyvoice

import sys
sys.path.append('third_party/Matcha-TTS')


from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)


from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
"""
import sounddevice as sd

devices = sd.query_devices()
if len(devices) == 0:
    print("No microphone devices found")
    sys.exit(0)
print(devices)
input_device_idx = sd.default.device[0]
print("Use this mic: %s" % devices[input_device_idx]["name"])
"""

def generate_voice(output):
    for i, j in enumerate(output):
        print(i)
        #torchaudio.save('zero_shot_{}_laoban.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        # 只支持macOS
        #torchaudio.io.play_audio(j['tts_speech'], cosyvoice.sample_rate, device=0)

        #https://python-sounddevice.readthedocs.io/en/0.5.1/usage.html#playback

        #print(type(j["tts_speech"]), j["tts_speech"].shape)
        # <class 'torch.Tensor'> torch.Size([1, 222720])
        sd.play(j['tts_speech'].cpu().detach().numpy().T, cosyvoice.sample_rate) # 24000 Hz
        sd.wait() # wait till the playback is complete
        # sd.stop() # will make it stop mid play

def save_voice(output):
    for i, j in enumerate(output):
        print(i)
        torchaudio.save('laopo_output.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./test_audio/laopo2.wav', 16000)

prompt_speech_text = "现在我们有很多突出的矛盾，比如说人岗不匹配，比如说这个整个学科设置不合理，那么就整个会导致我们培养出来的学生的能力，和真正的市场需求，他是脱节的。那么这个问题为什么会产生呢，一方面是因为现在整个科技的发展在加速，导致整个用工市场，对能力的需求的结构，也是在快速地变化。"

#cosyvoice = CosyVoice2('./pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=True, fp16=True,
#    prompt_text=prompt_speech_text, prompt_speech_16k=prompt_speech_16k)

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B',
    load_jit=False, load_trt=True, load_vllm=True, fp16=True,
    prompt_text=prompt_speech_text, prompt_speech_16k=prompt_speech_16k, gpu_memory_utilization=0.4)

# 微信语音，然后用苹果电脑quicktime录声音，然后转
# ffmpeg -i /Users/junweiliang/Downloads/zero_shot_prompt_laopo.m4a -ss 00:00:03 -to 00:00:06 -acodec pcm_s16le -ac 1 -ar 16000 zero_shot_prompt_laopo.wav
# if you want to slow down the audio: -filter:a "atempo=0.8"

# laopo2好像 好一些

trys = 3
for i in range(trys):
    output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
        prompt_speech_text, prompt_speech_16k, stream=False) # stream=True 下面才会有多个segment，效果很差，会卡

    #generate_voice(output)
    save_voice(output)

    output = cosyvoice.inference_zero_shot_fast('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
        stream=False) # stream=True 下面才会有多个segment，效果很差，会卡

    save_voice(output)

    print("--------- try again---------")
