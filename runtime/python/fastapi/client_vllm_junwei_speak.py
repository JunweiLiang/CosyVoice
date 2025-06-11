# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import requests
import torch
import torchaudio
import numpy as np
import time
import sounddevice as sd

devices = sd.query_devices()
if len(devices) == 0:
    print("No microphone devices found")
    sys.exit(0)
print(devices)
input_device_idx = sd.default.device[0]
print("Use this mic: %s" % devices[input_device_idx]["name"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='127.0.0.1')
    parser.add_argument('--port',
                        type=int,
                        default='50000')

    parser.add_argument('--tts_text',
                        type=str,
                        default='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。')


    args = parser.parse_args()

    url = "http://{}:{}/inference_zero_shot_fast".format(args.host, args.port)

    target_sr = 24000  # cosyvoice.sample_rate

    payload = {
        'tts_text': args.tts_text,
    }
    start_time = time.perf_counter()
    response = requests.request("GET", url, data=payload, stream=True)

    tts_audio = b''
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    #print(tts_speech.shape) #torch.Size([1, 287040])
    sd.play(tts_speech, target_sr) # 24000 Hz

    print("took %.3f seconds" % (time.perf_counter() - start_time))
    sd.wait() # wait till the playback is complete
