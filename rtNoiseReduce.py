import math
import time
import wave

import noisereduce as nr
import numpy as np
import pyaudio

RATE = 44100
FORMAT = pyaudio.paInt16
CHUNK = 2048
WIDTH = 2
THRESH = -55
NOISE_LEN = 16
RECORD_SECONDS = 5
WIN_LENGTH = CHUNK // 2
HOP_LENGTH = CHUNK // 4


def int16_to_float32(data):
    if np.max(np.abs(data)) > 32768:
        raise ValueError("Data has values above 32768")
    return (data / 32768.0).astype("float32")


def float32_to_int16(data):
    if np.max(data) > 1:
        data = data / np.max(np.abs(data))
    return np.array(data * 32767).astype("int16")


def np_audioop_rms(data, width):
    if len(data) == 0:
        return None

    fromType = (np.int8, np.int16, np.int32)[width // 2]
    d = np.frombuffer(data, fromType).astype(np.float)
    rms = np.sqrt(np.mean(d**2))

    return int(rms)


def main():
    aud = pyaudio.PyAudio()
    stream = aud.open(format=FORMAT, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

    noise = []
    frames = []
    unpFrames = []
    times = []

    print("recording")
    print("Processing (per sample) needs to be done in less than {} miliseconds".format(1000*CHUNK/RATE))
    sTime = time.time()
    lThresh = time.time() - 1
    avgQuiet = -60

    while time.time() - sTime < RECORD_SECONDS:
        data = stream.read(CHUNK)

        level = np_audioop_rms(data, WIDTH)
        if level:
            level = 20 * math.log10(level) - 100
        else:
            level = -101

        # gather the samples that represent noise
        # always be updating in case of bad samples
        if level < THRESH:
            if time.time() - lThresh > 1:  # time to delay after the last time the threshold was reached before stopping transmission
                avgQuiet = (avgQuiet * (NOISE_LEN - 1) + level) / NOISE_LEN  # moving average of the volume of the noise
                if level < avgQuiet:
                    noise.append(data)
                    if len(noise) > NOISE_LEN:
                        noise.pop(0)
        else:
            lThresh = time.time()

        unpFrames.append(data)

        if noise:
            tim = time.perf_counter()

            data = np.frombuffer(data, np.int16)
            data = int16_to_float32(data)
            nData = int16_to_float32(np.frombuffer(b''.join(noise), np.int16))

            data = nr.reduce_noise(audio_clip=data, noise_clip=nData,
                                   verbose=False, n_std_thresh=1.5, prop_decrease=1,
                                   win_length=WIN_LENGTH, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
                                   n_grad_freq=4)

            data = float32_to_int16(data)
            data = np.ndarray.tobytes(data)
            times += [1000*(time.perf_counter()-tim)]

        frames.append(data)

    times.pop(0)  # first one is always an outlier

    avg = sum(times)/len(times)
    print("Average processing time was {} miliseconds which is {}x faster than the min required".format(avg, 1000*(CHUNK/RATE) / avg))
    print("Worst time was {} miliseconds which is {}x faster than the min required".format(max(times), 1000*(CHUNK/RATE) / max(times)))

    stream.write(b''.join(unpFrames))  # play the unprocessed frames back

    time.sleep(.5)

    stream.write(b''.join(frames))  # play the processed frames back

    time.sleep(.5)

    post = b''.join(unpFrames)
    post = np.frombuffer(post, np.int16)
    post = int16_to_float32(post)
    nData = int16_to_float32(np.frombuffer(b''.join(noise), np.int16))
    post = nr.reduce_noise(audio_clip=post, noise_clip=nData,
                           verbose=False, n_std_thresh=1.5, prop_decrease=1,
                           win_length=WIN_LENGTH, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
                           n_grad_freq=4)

    post = float32_to_int16(post)
    post = np.ndarray.tobytes(post)

    stream.write(post)  # test to compare the whole thing processed at once

    # save the processed frames to a wav file for future reference
    waveFile = wave.open("test.wav", 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(aud.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


if __name__ == "__main__":
    main()
