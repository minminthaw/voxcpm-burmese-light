# OpenAI-Compatible Speech API Documentation

VoxCPM2 Lightweight provides a 100% drop-in replacement for the **OpenAI Audio Speech API** (`/v1/audio/speech`). Any application, SDK, or framework designed for OpenAI's TTS can immediately generate Burmese and English speech by changing the `base_url`.

---

## 1. API Endpoint Specification

* **Endpoint:** `POST /v1/audio/speech`
* **Content-Type:** `application/json`
* **Response Content-Type:** `audio/wav` or `audio/mpeg` (binary stream)

### Request JSON Parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `input` | `string` | **Yes** | — | The text to synthesize (Burmese, English, or mixed bilingual). |
| `model` | `string` | No | `"voxcpm2"` | Model identifier (`voxcpm2`, `tts-1`, or `tts-1-hd`). |
| `voice` | `string` | No | `"auto"` | Voice selection (`"auto"` for default, or path to a reference WAV file for zero-shot voice cloning). |
| `response_format` | `string` | No | `"wav"` | Audio format: `"wav"` (uncompressed 48kHz studio audio) or `"mp3"`. |
| `speed` | `float` | No | `1.0` | Speaking speed multiplier. |

---

## 2. Client Code Examples

### A. cURL

```bash
# Basic Burmese speech synthesis (WAV)
curl -X POST http://localhost:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2",
    "input": "မင်္ဂလာပါခင်ဗျာ။ ဒါကတော့ OpenAI API ကနေတဆင့် ထွက်ပေါ်လာတဲ့ အသံဖြစ်ပါတယ်။",
    "response_format": "wav"
  }' \
  --output speech.wav
```

---

### B. Python (Official OpenAI SDK)

```python
from openai import OpenAI

# Point to your local or Ubuntu server IP:
client = OpenAI(
    base_url="http://localhost:7860/v1",
    api_key="not-needed" # Local server doesn't require a key
)

response = client.audio.speech.create(
    model="voxcpm2",
    voice="auto",
    input="မင်္ဂလာပါခင်ဗျာ၊ ကျွန်တော်ကတော့ AI Assistant တစ်ယောက် ဖြစ်ပါတယ်။",
    response_format="wav"
)

# Stream directly to file
response.stream_to_file("output.wav")
print("Audio saved to output.wav (48,000 Hz)")
```

---

### C. Node.js / TypeScript (Official OpenAI SDK)

```typescript
import fs from "fs";
import path from "path";
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:7860/v1",
  apiKey: "not-needed",
});

async function main() {
  const speechFile = path.resolve("./speech.wav");

  const response = await openai.audio.speech.create({
    model: "voxcpm2",
    voice: "auto",
    input: "မင်္ဂလာပါခင်ဗျာ။ Node.js server ကနေ အောင်မြင်စွာ ခေါ်ယူလိုက်ပါပြီ။",
    response_format: "wav",
  });

  const buffer = Buffer.from(await response.arrayBuffer());
  await fs.promises.writeFile(speechFile, buffer);
  console.log("Saved 48kHz audio to:", speechFile);
}

main();
```

---

### D. Next.js / Browser (Native `fetch`)

```javascript
// Example Next.js Route Handler or Frontend call
export async function POST(req) {
  const { text } = await req.json();

  const response = await fetch("http://localhost:7860/v1/audio/speech", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "voxcpm2",
      input: text,
      response_format: "wav",
    }),
  });

  const audioBlob = await response.blob();
  return new Response(audioBlob, {
    headers: { "Content-Type": "audio/wav" },
  });
}
```

---

### E. PHP (cURL / Laravel)

```php
<?php

$payload = json_encode([
    'model' => 'voxcpm2',
    'input' => 'မင်္ဂလာပါခင်ဗျာ။ PHP application ကနေ အသံထုတ်ယူခြင်း ဖြစ်ပါတယ်။',
    'response_format' => 'wav'
]);

$ch = curl_init('http://localhost:7860/v1/audio/speech');
curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "POST");
curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: application/json',
    'Content-Length: ' . strlen($payload)
]);

$result = curl_exec($ch);
curl_close($ch);

file_put_contents('php_speech.wav', $result);
echo "Audio saved to php_speech.wav\n";
```

---

## 3. Zero-Shot Voice Cloning via API

To clone any speaker's voice via the API, pass the path to the reference WAV in the `voice` field:

```bash
curl -X POST http://localhost:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2",
    "voice": "examples/reference_speaker.wav",
    "input": "ဒီအသံကတော့ စပီကာရဲ့ အသံနမူနာကို ပုံတူကူးယူထားတဲ့ အသံဖြစ်ပါတယ်။"
  }' \
  --output cloned_speech.wav
```

---

## 4. Health Check Endpoint

```bash
curl -s http://localhost:7860/health | jq
```

Response:
```json
{
  "status": "healthy",
  "device": "mps",
  "model_loaded": true,
  "active_mode": "int8"
}
```
