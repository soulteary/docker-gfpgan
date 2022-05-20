# Docker GFPGAN

TencentARC GFPGAN ALL-IN-ONE Docker Images.

support for x86 and ARM (M1, M1Pro).

## How to Use

It's easy to use:

```bash
docker pull soulteary/docker-gfpgan
```

Mount the `model` and the `images` to be processed to the container:

```bash
docker run --rm -it -v `pwd`/model/GFPGANCleanv1-NoCE-C2.pth:/GFPGAN.pth -v `pwd`/data:/data soulteary/docker-gfpgan:2022.05.20

# Or with version
# docker run --rm -it -v `pwd`/model/GFPGANCleanv1-NoCE-C2.pth:/GFPGAN.pth -v `pwd`/data:/data soulteary/docker-gfpgan:2022.05.20
```

Take a sip of water, wait a few seconds, the magic will appear (`./data/result.html`):

![](./screenshots/preview.png)

## Docker Images

- soulteary/docker-gfpgan:latest
- soulteary/docker-gfpgan:2022.05.20

## Build

```bash
docker build -t soulteary/docker-gfpgan .
```

## Refs

- https://github.com/TencentARC/GFPGAN
- https://github.com/soulteary/docker-pytorch-playground
