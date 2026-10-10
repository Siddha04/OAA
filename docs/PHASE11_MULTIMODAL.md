# OAA Phase 11 - Local Vision and Voice Foundation

## Included

- Decode one selected local JPEG, PNG, or WebP file to orientation-corrected RGB bytes.
- Enforce a maximum image byte count, pixel count, and longest side; rescale with a high-quality filter when needed.
- Read bounded, uncompressed local PCM WAV files with mono/stereo channels and 8–192 kHz sample rates.
- Normalize integer PCM samples to floating point in [-1, 1] without recording from a microphone.
- Define explicit SpeechToTextBackend and TextToSpeechBackend protocols for integrating local models.
- Validate generated PCM output and write it as WAV, refusing to overwrite an existing file by default.
- Add CLI input inspection without invoking a model: genai media image-info and genai media audio-info.

## Usage

Install Pillow for image decoding:

    python -m pip install "oaa[vision]"

Inspect an image:

    genai media image-info --path ./photo.png

Inspect an audio clip:

    genai media audio-info --path ./speech.wav

Python example:

    from oaa.media import load_image, load_wav

    image = load_image("./photo.png")
    print(image.summary())
    # image.rgb_bytes contains row-major RGB data suitable for a future vision encoder

    audio = load_wav("./speech.wav")
    samples = audio.iter_normalized_samples()
    print(audio.summary())

A local ASR or TTS backend can be supplied explicitly to transcribe_audio(audio, backend) or synthesize_speech(text, backend). No ASR/TTS provider is registered by default. The library never selects a cloud provider implicitly.

## Safety bounds and limitations

- Image files are size-limited and pixel-limited before the RGB frame is returned; unsupported formats are rejected.
- WAV files must be uncompressed PCM, within the configured file/duration limits, and have valid sample rates/channel counts.
- No URLs, camera, or microphone are accessed.
- This is input preparation and adapter plumbing, not a working image-understanding or voice conversation model. No pretrained vision encoder, OCR, ASR, or TTS implementation is bundled.
- The current tiny text transformer remains text-only; multimodal reasoning requires a compatible encoder/model and runtime integration in a later phase.
