from cottage_analysis.decorators import test_func


def test_slurm_my_func():
    # usage:
    # not using slurm
    out = test_func(1, 2, use_slurm=False)
    assert out == 3
    # using slurm
    out = test_func(1, 2, slurm_folder=".")
    assert isinstance(out, str)
    # wait for previous job to finish
    test_func(1, 2, scripts_name="test_func_renamed", slurm_folder=".")
    # rename the slurm script
    test_func(
        1, 2, scripts_name="test_func_with_dep", job_dependency=out, slurm_folder="."
    )


if __name__ == "__main__":
    test_slurm_my_func()
