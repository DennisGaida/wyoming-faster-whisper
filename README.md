# Wyoming Faster Whisper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [faster-whisper](https://github.com/guillaumekln/faster-whisper/) speech to text system.

## Home Assistant Add-on

[![Show add-on](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=core_whisper)

[Source](https://github.com/home-assistant/addons/tree/master/whisper)

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/rhasspy/wyoming-faster-whisper.git
cd wyoming-faster-whisper
script/setup
```

Run a server anyone can connect to:

```sh
script/run --model tiny-int8 --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data
```

The `--model` can also be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

**NOTE**: Models are downloaded to the first `--data-dir` directory.

## Biasing Toward Your Home Assistant Names

Whisper has never heard of your thermostat. "What's the temperature of the Ecobee?"
comes back as "What's the temperature of the incubi?" — the acoustics were fine,
the model just has no reason to think that word exists.

Given a long-lived access token, the server reads the names in your home over the
Home Assistant websocket API and feeds them to the model as a prompt, which fixes
exactly that class of error:

```sh
script/run --uri 'tcp://0.0.0.0:10300' --data-dir /data \
    --hass-token "$TOKEN" --hass-api 'http://homeassistant.local:8123/api'
```

Requires the `hass` extra:

```sh
pip install 'wyoming-faster-whisper[hass]'
```

It collects the names of **conversation-exposed** entities and their aliases, plus
your area and floor names — the names a speaker can actually say. Nothing else is
read, and no service is ever called.

The fetch is free in latency terms: it starts when the audio starts, while the
speaker is still talking, and the names are ready by the time the audio stops.
Home Assistant being slow or unreachable only costs freshness — the previous names
are used, or none at all, and the transcript still comes back.

| Option | Default | Purpose |
| --- | --- | --- |
| `--hass-token` | | Long-lived access token. Enables everything above. |
| `--hass-api` | `http://homeassistant.local:8123/api` | Where to find Home Assistant. |
| `--hass-refresh-seconds` | `0` | Minimum seconds between refreshes. `0` refreshes every utterance, so a rename takes effect immediately. |
| `--hass-prompt-max-tokens` | `200` | Token budget for names. Whisper's hard cap is 223 and quality falls off before it. |
| `--hass-prompt-timeout` | `1.0` | How long to wait on an unfinished refresh before transcribing with the names already on hand. |

A large home has more names than the budget holds. They are added in priority
order — areas, floors, entity names, then aliases — and cut off when the budget
runs out; run with `--debug` to see how many were dropped and the exact prompt
used. `--initial-prompt` still works and is kept at the front of the prompt, ahead
of anything discovered from Home Assistant.

This biases `faster-whisper` and `qwen3-asr`, the backends that take a prompt.
Others ignore it.

## Docker Image

``` sh
docker run -it -p 10300:10300 -v /path/to/local/data:/data rhasspy/wyoming-whisper \
    --model tiny-int8 --language en
```

**NOTE**: Models are downloaded to `/data`, so make sure this points to a Docker volume.

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/whisper)
