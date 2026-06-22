"""Bundle lambdas/ and tsf2_core/ for Lambda deployment assets."""

from __future__ import annotations

import shutil
from pathlib import Path

import jsii
from aws_cdk import BundlingOptions, ILocalBundling
from aws_cdk import aws_lambda as _lambda

ROOT = Path(__file__).resolve().parents[1]


@jsii.implements(ILocalBundling)
class _LambdaBundleLocal:
    def try_bundle(self, output_dir: str, _options) -> bool:
        output = Path(output_dir)
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        shutil.copytree(ROOT / "lambdas", output, dirs_exist_ok=True)
        shutil.copytree(ROOT / "tsf2_core", output / "tsf2_core", dirs_exist_ok=True)
        return True


def lambda_asset_code() -> _lambda.Code:
    return _lambda.Code.from_asset(
        str(ROOT),
        exclude=[
            "*",
            "!lambdas",
            "!lambdas/**",
            "!tsf2_core",
            "!tsf2_core/**",
        ],
        bundling=BundlingOptions(
            image=_lambda.Runtime.PYTHON_3_12.bundling_image,
            command=[
                "bash",
                "-c",
                "mkdir -p /asset-output && "
                "cp -r /asset-input/lambdas/. /asset-output/ && "
                "cp -r /asset-input/tsf2_core /asset-output/tsf2_core",
            ],
            local=_LambdaBundleLocal(),
        ),
    )
