# Data

## Source

SMS Spam Collection from the UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

5574 English SMS messages, each labelled `ham` (normal) or `spam`.
We downloaded the zip and put the file `SMSSpamCollection` in `data/raw/`.
Each line is `label<TAB>message`.

What we found when we explored it (`notebooks/explore.ipynb`):

- 4827 ham (86.6%) and 747 spam (13.4%), so the classes are not balanced
- 403 messages appear more than once (exact duplicates)
- spam messages are longer on average (about 139 characters vs 71 for ham) and often
  contain words like free, win, call, txt, and phone numbers

## Versions

Both files are tracked with DVC and stored on Azure (remote `myazure`).
Git only has the small `.dvc` files.

| Version | File | Rows | How it is made |
|---|---|---|---|
| raw | `data/raw/SMSSpamCollection` | 5574 | downloaded from UCI |
| v1 | `data/sms.csv` | 5574 | `python ml/data_pipeline.py --version v1` (raw data, nothing removed) |
| v2 | `data/sms.csv` | 5159 | `python ml/data_pipeline.py --version v2` |

### What changed from v1 to v2

- text in lower case and extra spaces / tabs / new lines removed
- duplicates removed (403 exact duplicates + 12 more that only differed in case or
  spaces, 415 in total)
- empty messages removed (there were none, but we keep the check)

Why: with duplicates, the same message can end up in the training set and in the test
set. In v1, 135 messages of the test set are also in the training set, so the score is
too optimistic (data leakage). In v2 the number is 0. v2 gives a more honest score.

Both numbers come from the same split the training script uses
(`test_size=0.2, random_state=42, stratify=y`):

| Data | Rows | Train | Test | Test messages also in train |
|---|---|---|---|---|
| v1 | 5574 | 4459 | 1115 | 135 |
| v2 | 5159 | 4127 | 1032 | 0 |

## Code <-> data

| Version | Git tag | Commit | DVC md5 of `data/sms.csv` |
|---|---|---|---|
| v1 | `data-v1` | `bc2b403` | `3819cc6774c336bd7114795a31e8b268` |
| v2 | `data-v2` | `4230266` | `4f2c7c3054fbd9a2970296c932f15fb6` |

Get a version back:

```bash
git checkout data-v1 -- data/sms.csv.dvc
dvc pull data/sms.csv
```

### Before you can pull

`dvc pull` needs the Azure SAS token, and the token is **not** in Git: it lives in
`.dvc/config.local`, which DVC keeps out of version control. On a fresh clone, set it up
once:

```bash
dvc remote modify --local myazure sas_token '<token>'
dvc pull
dvc status -c        # Cache and remote 'myazure' are in sync.
```

To check that the remote is configured, use `dvc remote list`. Do **not** use
`dvc remote modify --local myazure sas_token` without a value to read the token back: on
DVC 3.67.1 that form fails with
`configuration error: expected str for dictionary value`. The token is still written and
used correctly — `dvc push`, `dvc pull` and `dvc status -c` all work — only the
read-back query is broken. The same error also appears for a plain value such as
`account_name`, which is how we know it is not a problem with the token itself.

The md5 is what links a trained model to the data it used: `ml/train.py` reads it from
the `.dvc` pointer file and logs it to MLflow as the `dvc_md5` tag.
