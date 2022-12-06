# Paraphrase-Detector

The paraphrase detector checks the likelihood of the 2 given sentences being a paraphrase of each other.

## Usage

Clone the repository

```shell
python3 -m flask run
```

You can then get predictions by CURL or the Web UI at [127.0.0.1:5000](http://127.0.0.1:5000/).

```shell
curl -X POST -H 'Content-Type: application/json' -d '{"sent_1": "Can you recommend some upscale restaurants in New York?", "sent_2": "List some excellent restaurants to visit in New York City?"}' localhost:5000/predict
```