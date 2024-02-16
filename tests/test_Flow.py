from shelephant.path import cwd

from depinning_inertia_2024 import Flow


def dict2list(d):
    ret = [[k, v] for k, v in d.items()]
    return [item for row in ret for item in row]


def test_basic(tmp_path):
    Flow.cli_generate(
        dict2list(
            {
                "--eta": 1e0,
                "--size": 50,
                "-n": 1,
                "--v-frame": 1,
                "--kframe": 1 / 50,
                "--nstep": 100,
                "--dev": tmp_path,
            }
        )
    )

    with cwd(tmp_path):
        Flow.cli_run(["--dev", "id=0000.h5"])
        Flow.cli_ensemblepack(["--dev", "-o", "info.h5", "id=0000.h5"])
        Flow.cli_ensemblepack(["--dev", "-o", "info.h5", "-i", "id=0000.h5"])


def test_thermal(tmp_path):
    Flow.cli_generate(
        dict2list(
            {
                "--eta": 1e0,
                "--size": 50,
                "-n": 1,
                "--v-frame": 1,
                "--kframe": 1 / 50,
                "--nstep": 100,
                "--temperature": 0.1,
                "--dev": tmp_path,
            }
        )
    )

    with cwd(tmp_path):
        Flow.cli_run(["--dev", "id=0000.h5"])
