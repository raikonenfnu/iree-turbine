# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
import torch

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


def launch(func: Callable[[], None]) -> Callable[[], None]:
    """
    Run a function as part of the test suite in a test launch context.
    Provides default values for the hyperparameters.
    """
    if __name__ == "__main__":
        with tk.gen.TestLaunchContext(
            {
                M: 16,
                N: 16,
                K: 16,
                BLOCK_M: 16,
                BLOCK_N: 16,
                BLOCK_K: 16,
                ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            }
        ):
            func()
    return func


def run(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    if __name__ == "__main__":
        func()
        # Print a separator between tests
        print("-----")
    return func


def codegen_test_context():
    return tk.gen.TestLaunchContext(
        {
            M: 16,
            N: 16,
            K: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            BLOCK_K: 16,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        }
    )


@run
def test_read():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    @tkw.wave(constraints)
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.read(a, elements_per_thread=4)

    with codegen_test_context():
        a = torch.randn(16, 16, dtype=torch.float16)
        print(test(a).module_op)
        # CHECK: func.func @test(%[[ARG0:.+]]: !stream.binding)
        # CHECK: %[[WG_0:.+]] = stream.dispatch.workgroup.id[0]
        # CHECK: %[[WG_1:.+]] = stream.dispatch.workgroup.id[1]
        # CHECK: %[[DATA:.+]] = stream.binding.subspan %[[ARG0]]
        # CHECK: %[[C16:.+]] = arith.constant 16 : index
        # CHECK: %[[IDX_X:.+]] = arith.muli %[[WG_0]], %[[C16]]
        # CHECK: %[[C16_1:.+]] = arith.constant 16 : index
        # CHECK: %[[IDX_Y:.+]] = arith.muli %[[WG_1]], %[[C16_1]]
        # CHECK: vector.load %[[DATA]][%[[IDX_X]], %[[IDX_Y]]] : memref<16x16xf16>, vector<4xf16>


@run
def test_add():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    @tkw.wave(constraints)
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a_reg = tkw.read(a, elements_per_thread=4)
        res = a_reg + a_reg

    with codegen_test_context():
        a = torch.randn(16, 16, dtype=torch.float16)
        print(test(a).module_op)
        # CHECK: func.func @test(%[[ARG0:.+]]: !stream.binding)
        # CHECK: %[[WG_0:.+]] = stream.dispatch.workgroup.id[0]
        # CHECK: %[[WG_1:.+]] = stream.dispatch.workgroup.id[1]
        # CHECK: %[[DATA:.+]] = stream.binding.subspan %[[ARG0]]
        # CHECK: %[[C16:.+]] = arith.constant 16 : index
        # CHECK: %[[IDX_X:.+]] = arith.muli %[[WG_0]], %[[C16]]
        # CHECK: %[[C16_1:.+]] = arith.constant 16 : index
        # CHECK: %[[IDX_Y:.+]] = arith.muli %[[WG_1]], %[[C16_1]]
        # CHECK: %[[SLICE:.+]] = vector.load %[[DATA]][%[[IDX_X]], %[[IDX_Y]]] : memref<16x16xf16>, vector<4xf16>
        # CHECK: arith.addf %[[SLICE]], %[[SLICE]] : vector<4xf16>


@launch
@pytest.mark.skip(reason="neg: Currently only stub implementation")
def test_neg():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    @tkw.wave(constraints)
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        res = -a
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    with pytest.raises(
        NotImplementedError, match="neg: Currently only stub implementation"
    ):
        test(a)


@launch
@pytest.mark.skip(reason="sub: Currently only stub implementation")
def test_sub():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    @tkw.wave(constraints)
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        res = a - a
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    with pytest.raises(
        NotImplementedError, match="sub: Currently only stub implementation"
    ):
        test(a)


@launch
@pytest.mark.skip(reason="getitem: Currently only stub implementation")
def test_get_item():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        ),
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    @tkw.wave(constraints)
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        res = a[0]
        tkw.write(res, a, elements_per_thread=4)

    a = torch.randn(16, 16, dtype=torch.float16)
    with pytest.raises(
        NotImplementedError, match="getitem: Currently only stub implementation"
    ):
        test(a)


# TODO: Add more tests once we have more than a stub implementation.
