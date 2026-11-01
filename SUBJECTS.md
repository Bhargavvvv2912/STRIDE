# Subjects

The sample, and why each repository is in it. Regenerate with
`python scripts/subject_status.py`.

A subject needs a working environment to measure against -- its own
requirements, or the reconstruction of them at its commit date -- and its
own tests. It does *not* need the naive upgrade to break: requiring that
would select exactly the cases where the baselines fail, so subjects where
upgrading everything already works are kept, and every arm is expected to
tie on them. An arm that breaks such a subject is worth knowing about.

| subject | era | repository | tests | python | baseline | naive upgrade | runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean-review-collector | recent | Neo-Hao/mean-review-collector | 19 | 3.9 | as written | breaks | done (11 arms) |
| flask-dj | stale | AlexandrovRoman/Flask-DJ | 24 | 3.9 | reconstructed (28 pkgs) | breaks | done (11 arms) |
| test-flask | stale | rafegal/test-flask | 5 | 3.9 | reconstructed (10 pkgs) | passes (tie expected) | done (11 arms) |
| sayhup | stale | boxoforanmore/SayHup | 9 | 3.9 | reconstructed (10 pkgs) | passes (tie expected) | done (11 arms) |
| swiftenv-api | recent | kylef/swiftenv-api | 6 | 3.9 | as written | passes (tie expected) | done (11 arms) |
| term-image | current | AnonymouX47/term-image | 1210 | 3.9 | as written | breaks | done (11 arms) |
| scanvix | current | ASWINMS07/ScanVIX | 152 | 3.13 | as written | passes (tie expected) | done (11 arms) |
| find-that-postcode | current | kanedata/find-that-postcode | 47 | 3.12 | as written | passes (tie expected) | done (11 arms) |
| linebot-test | current | brian0913579/linebot.test | 140 | 3.13 | as written | passes (tie expected) | done (11 arms) |
| lung-cancer-detection | current | nordszamora/lung-cancer-detection | 14 | 3.13 | as written | breaks | done (11 arms) |
| pylesa | current | andrewlyden/PyLESA | 91 | 3.12 | as written | passes (tie expected) | not yet run |
| age-of-empires-api | current | aalises/age-of-empires-II-api | 7 | 3.12 | as written | passes (tie expected) | done (11 arms) |
| sms-code-sync | current | hxabcd/sms-code-sync | 5 | 3.13 | as written | passes (tie expected) | done (11 arms) |
| missing-person-detector | current | Arjun-ys/Missing-Person-Detector | 5 | 3.11 | reconstructed (6 pkgs) | passes (tie expected) | done (11 arms) |
| nba2k25-plp | current | JonasDockx/NBA2K25_PLP | 199 | 3.13 | as written | passes (tie expected) | done (11 arms) |
| pytorch-ocr | recent | GabrielDornelles/pytorch-ocr | 44 | 3.10 | as written | passes (tie expected) | done (11 arms) |
| flask-restful-example | stale | erdem/flask-restful-example | 13 | 3.9 | reconstructed (15 pkgs) | breaks | done (11 arms) |
| pacman | current | Shtomuch/PacMan | 102 | 3.13 | as written | passes (tie expected) | done (11 arms) |
| hackton-intgest | current | Xamu-192/HACKTON-2026-INTGEST | 170 | 3.13 | as written | passes (tie expected) | done (11 arms) |
| lambda-chaos-injection | current | adhorn/aws-lambda-chaos-injection | 17 | 3.8 | as written | passes (tie expected) | done (11 arms) |
| fork-recipes | current | mikebgrep/fork.recipes | 29 | 3.12 | as written | passes (tie expected) | done (11 arms) |
| htb-api | recent | clubby789/htb-api | 52 | 3.10 | as written | breaks | not yet run |

## Strata

Dependency rot is a function of time, so the sample is stratified by
the date of each subject's pinned commit. A stratum with no subjects
is a gap in the sample, not an absence of projects.

| era | years | what it means | subjects |
| --- | --- | --- | --- |
| current | 2024+ | pins are still close to the releases they name | 14 |
| recent | 2021-2023 | Flask 2, setuptools 60 and the end of easy_install | 4 |
| stale | 2018-2020 | Python 3.7-3.8, Flask 1.x, before the metadata rewrites | 4 |
| old | 2015-2017 | Python 3.5-3.6, before wheels were universal | 0 |
| ancient | 1990-2014 | the Python 2 era; most of these no longer install at all | 0 |

## Screened and not used (54 of 78)

Kept as a record rather than deleted: how often a published Python
project can no longer be installed or tested is itself a result, and it
is the denominator behind every claim about the sample.

- `64Sanjay/biometric-user-authentication-using-Blockchain` -- no working baseline on 3.10, 3.9, 3.8: 
- `Baxter-College/baxtabot` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: ImportError: cannot import name 'Callable' from 'coll
- `Boeun-Kim/GL-Transformer` -- no working baseline on 3.10, 3.9: E   ImportError: libGL.so.1: cannot open shared object file: No such file or
- `DeviPravallikaKarry/expense-management-system` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: E   ModuleNotFoundError: No module named 'logging_set
- `Ernisius/Picochess-NonDGT_PI` -- no tests found
- `FlaskAio/navycut` -- no working baseline on 3.10, 3.9, 3.8: 
- `FoxML-JENNIFER/Aurora` -- no working baseline on 3.10, 3.9, 3.8: E   ImportError: cannot import name 'UTC' from 'datetime' (/usr/local/l
- `IDSIA/sacred` -- screening crashed: UnboundLocalError: cannot access local variable 'app' where it is not associated with a val
- `IEEE-VIT/flask.ly` -- no tests found
- `MuongKimhong/toschat` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: E               ConnectionRefusedError: [Errno 111] C
- `ScriptLineStudios/Imagscii` -- no tests found
- `YangCatalog/backend` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: CompileError: src/gevent/libev/corecext.pyx
- `Yun-cong/DoubanAPI` -- no tests found
- `adityarajesh010/RTDLS-Project` -- no working baseline on 3.10, 3.9, 3.8: No matching distribution found for sqlite3
- `arnoob16/cpAPI` -- no tests found
- `azimjohn/musicspider` -- no tests found
- `benfred/implicit` -- screening crashed: UnboundLocalError: cannot access local variable 'app' where it is not associated with a val
- `bodegafresh/md2pdf` -- no working baseline on 3.11, 3.13, 3.12, 3.10, 3.9, 3.8: 
- `brenokcc/djangoplus` -- no working baseline on 3.10, 3.9, 3.8: E   ModuleNotFoundError: No module named 'djangoplus'
- `buildingSMART/ifc-gherkin-rules` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: 
- `carbonblack/cb-taxii-connector` -- no working baseline on 3.10, 3.9, 3.8: No matching distribution found for cbfeeds==1.0.0
- `cloudacademy/fraud-detection` -- no working baseline on 3.10, 3.9, 3.8: No matching distribution found for pandas==0.20.3
- `colindecourt/record` -- no working baseline on 3.10, 3.9, 3.8: E   ModuleNotFoundError: No module named 'cruw'
- `denisvasilik/binalyzer` -- no working baseline on 3.10, 3.9, 3.8: 
- `erent8/opencv-doc-TR` -- no working baseline on 3.10, 3.9, 3.8: E       fixture 'module_name' not found
- `fluxxset/Evilpunch` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: 
- `gae-init/gae-init-docs` -- no working baseline on 3.10, 3.9, 3.8: E   ModuleNotFoundError: No module named 'google'
- `hda-technical/dancebooks` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: E   FileNotFoundError: [Errno 2] No such file or dire
- `huogerac/flask-vercel` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: AttributeError: 'build_ext' object has no attribute '
- `hylinux1024/datingtoday` -- no working baseline on 3.10, 3.9, 3.8: E   ImportError: cannot import name 'Mapping' from 'collections' (/usr/
- `idealista/prom2teams` -- no working baseline on 3.10, 3.9, 3.8: 
- `im-syn/SafeVision` -- no working baseline on 3.10, 3.9, 3.8: E   ImportError: libGL.so.1: cannot open shared object file: No such fi
- `jeganathpv/flask-firebase-crud` -- no tests found
- `jhezjkp/flask-principal` -- no working baseline on 3.9, 3.8, 3.10: SyntaxError: Missing parentheses in call to 'print'. Did you mean print
- `mfreeborn/fastapi-sqlalchemy` -- no working baseline on 3.10, 3.9, 3.8: error: command '/usr/bin/gcc' failed with exit code 1
- `mjhea0/flask-basic-registration` -- no working baseline on 3.9, 3.8, 3.10: E   KeyError: 'APP_SETTINGS'
- `myaooo/RNNVis` -- no working baseline on 3.10, 3.9, 3.8: ModuleNotFoundError: No module named 'numpy'
- `noahgift/pai-aws` -- no working baseline on 3.9, 3.8, 3.10: E   ModuleNotFoundError: No module named 'web'
- `oauthlib/oauthlib` -- screening crashed: UnboundLocalError: cannot access local variable 'app' where it is not associated with a val
- `onuar/dokuztas` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: ImportError: cannot import name 'Feature' from 'setup
- `pavankumark24/malware-detector` -- no working baseline on 3.10, 3.9, 3.8: 
- `pokealarm/pokealarm` -- no working baseline on 3.10, 3.9, 3.8: E           AttributeError: 'function' object has no attribute 'info'
- `psf/requests` -- no requirements.txt
- `psi16181918161phi/LongFilePathScanner` -- no working baseline on 3.10, 3.9, 3.8: No matching distribution found for unittest>=3.9.0
- `rahmanfadhil/flask-rest-api` -- no tests found
- `rasmusbergpalm/evostrat` -- no working baseline on 3.10, 3.9, 3.8: RuntimeError: Running cythonize failed!
- `saurabhshri/ccextractor-web` -- no working baseline on 3.9, 3.8, 3.10: CalledProcessError: Command '['pkg-config', '--libs-only-L', 'libvirt']
- `sesam-community/sesam-py` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: 
- `shidenggui/easytrader` -- no working baseline on 3.10, 3.9, 3.8: No matching distribution found for pandas==0.23.3
- `specmatic/specmatic-python-extensions` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: E   FileNotFoundError: [Errno 2] No such file or dire
- `tanwanirahul/django-batch-requests` -- no working baseline on 3.13, 3.12, 3.11, 3.10, 3.9, 3.8: 
- `taoztw/Transformer` -- no tests found
- `toandeptrai16zz/remote-hardware-lab` -- no working baseline on 3.10, 3.9, 3.8: 
- `vjanz/flask-cache-redis` -- no tests found
