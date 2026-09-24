# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

import pytest
import os
import filecmp
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed.runtime.swap_tensor.utils import get_aio_aligned_bytes, AIO_ALIGNED_BYTES
from unit.common import DistributedTest

QUEUE_DEPTH = 2
IO_BLOCKS = 4
IO_PARALLEL = 2

if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
    pytest.skip('Skip tests since async-io is not compatible', allow_module_level=True)


def _skip_for_invalid_environment(use_cuda_pinned_tensor=True):
    if get_accelerator().device_name() != 'cuda':
        if use_cuda_pinned_tensor:
            pytest.skip("torch.pin_memory is only supported in CUDA environments.")


def _get_local_rank():
    if get_accelerator().is_available():
        return dist.get_rank()
    return 0


def _get_file_path(tmpdir, file_prefix, index=0):
    file_suffix = f'{_get_local_rank()}_{index}'
    return os.path.join(tmpdir, f'{file_prefix}_{file_suffix}.pt')


def _get_block_and_io_size(tmpdir):
    # O_DIRECT rejects I/O that is not aligned to the sector size of the tmpdir device (4096 on 4K-native disks),
    # so derive the block size, and hence every per-thread chunk, from the probed alignment.
    block_size = get_aio_aligned_bytes(str(tmpdir))
    return block_size, IO_BLOCKS * block_size


def _do_ref_write(tmpdir, num_bytes, index=0):
    ref_file = _get_file_path(tmpdir, '_py_random', index)
    ref_buffer = os.urandom(num_bytes)
    with open(ref_file, 'wb') as f:
        f.write(ref_buffer)

    return ref_file, ref_buffer


def _get_test_write_file(tmpdir, index):
    return _get_file_path(tmpdir, '_aio_write_random', index)


def _get_test_write_file_and_unpinned_tensor(tmpdir, ref_buffer, index=0):
    test_file = _get_test_write_file(tmpdir, index)
    test_buffer = get_accelerator().ByteTensor(list(ref_buffer))
    return test_file, test_buffer


def _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer, aio_handle=None, index=0):
    test_file = _get_test_write_file(tmpdir, index)
    if aio_handle is None:
        test_buffer = get_accelerator().pin_memory(torch.ByteTensor(list(ref_buffer)))
    else:
        tmp_buffer = torch.ByteTensor(list(ref_buffer))
        test_buffer = aio_handle.new_cpu_locked_tensor(len(ref_buffer), tmp_buffer)
        test_buffer.data.copy_(tmp_buffer)

    return test_file, test_buffer


def _validate_handle_state(handle, single_submit, overlap_events, block_size):
    assert handle.get_single_submit() == single_submit
    assert handle.get_overlap_events() == overlap_events
    assert handle.get_intra_op_parallelism() == IO_PARALLEL
    assert handle.get_block_size() == block_size
    assert handle.get_queue_depth() == QUEUE_DEPTH


@pytest.mark.parametrize("use_cuda_pinned_tensor", [True, False])
@pytest.mark.parametrize("single_submit", [True, False])
@pytest.mark.parametrize("overlap_events", [True, False])
class TestRead(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize("use_unpinned_tensor", [True, False])
    def test_parallel_read(self, tmpdir, use_cuda_pinned_tensor, single_submit, overlap_events, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        block_size, io_size = _get_block_and_io_size(tmpdir)

        h = AsyncIOBuilder().load().aio_handle(block_size, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        if use_unpinned_tensor:
            aio_buffer = torch.empty(io_size, dtype=torch.uint8, device=get_accelerator().device_name())
        elif use_cuda_pinned_tensor:
            aio_buffer = get_accelerator().pin_memory(torch.empty(io_size, dtype=torch.uint8, device='cpu'))
        else:
            aio_buffer = h.new_cpu_locked_tensor(io_size, torch.empty(0, dtype=torch.uint8))

        _validate_handle_state(h, single_submit, overlap_events, block_size)

        ref_file, _ = _do_ref_write(tmpdir, io_size)
        read_status = h.sync_pread(aio_buffer, ref_file, 0)
        assert read_status == 1

        with open(ref_file, 'rb') as f:
            ref_buffer = list(f.read())
        assert ref_buffer == aio_buffer.tolist()

        if not use_cuda_pinned_tensor:
            h.free_cpu_locked_tensor(aio_buffer)

    @pytest.mark.parametrize("use_unpinned_tensor", [True, False])
    def test_async_read(self, tmpdir, use_cuda_pinned_tensor, single_submit, overlap_events, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        block_size, io_size = _get_block_and_io_size(tmpdir)

        use_cpu_locked_tensor = False
        h = AsyncIOBuilder().load().aio_handle(block_size, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        if use_unpinned_tensor:
            aio_buffer = torch.empty(io_size, dtype=torch.uint8, device=get_accelerator().device_name())
        elif use_cuda_pinned_tensor:
            aio_buffer = get_accelerator().pin_memory(torch.empty(io_size, dtype=torch.uint8, device='cpu'))
        else:
            aio_buffer = h.new_cpu_locked_tensor(io_size, torch.empty(0, dtype=torch.uint8))
            use_cpu_locked_tensor = True

        _validate_handle_state(h, single_submit, overlap_events, block_size)

        ref_file, _ = _do_ref_write(tmpdir, io_size)
        read_status = h.async_pread(aio_buffer, ref_file, 0)
        assert read_status == 0

        wait_status = h.wait()
        assert wait_status == 1

        with open(ref_file, 'rb') as f:
            ref_buffer = list(f.read())
        assert ref_buffer == aio_buffer.tolist()

        if use_cpu_locked_tensor:
            h.free_cpu_locked_tensor(aio_buffer)


@pytest.mark.parametrize("use_cuda_pinned_tensor", [True, False])
@pytest.mark.parametrize("single_submit", [True, False])
@pytest.mark.parametrize("overlap_events", [True, False])
class TestWrite(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize("use_unpinned_tensor", [True, False])
    def test_parallel_write(self, tmpdir, use_cuda_pinned_tensor, single_submit, overlap_events, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        block_size, io_size = _get_block_and_io_size(tmpdir)

        ref_file, ref_buffer = _do_ref_write(tmpdir, io_size)
        h = AsyncIOBuilder().load().aio_handle(block_size, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        if use_unpinned_tensor:
            aio_file, aio_buffer = _get_test_write_file_and_unpinned_tensor(tmpdir, ref_buffer)
        if use_cuda_pinned_tensor:
            aio_file, aio_buffer = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer)
        else:
            aio_file, aio_buffer = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer, h)

        _validate_handle_state(h, single_submit, overlap_events, block_size)

        write_status = h.sync_pwrite(aio_buffer, aio_file, 0)
        assert write_status == 1

        if not use_cuda_pinned_tensor:
            h.free_cpu_locked_tensor(aio_buffer)

        assert os.path.isfile(aio_file)

        filecmp.clear_cache()
        assert filecmp.cmp(ref_file, aio_file, shallow=False)

    @pytest.mark.parametrize("use_unpinned_tensor", [True, False])
    def test_async_write(self, tmpdir, use_cuda_pinned_tensor, single_submit, overlap_events, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        block_size, io_size = _get_block_and_io_size(tmpdir)

        ref_file, ref_buffer = _do_ref_write(tmpdir, io_size)

        h = AsyncIOBuilder().load().aio_handle(block_size, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
        use_cpu_locked_tensor = False
        if use_unpinned_tensor:
            aio_file, aio_buffer = _get_test_write_file_and_unpinned_tensor(tmpdir, ref_buffer)
        elif use_cuda_pinned_tensor:
            aio_file, aio_buffer = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer)
        else:
            aio_file, aio_buffer = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer, h)
            use_cpu_locked_tensor = True

        _validate_handle_state(h, single_submit, overlap_events, block_size)

        write_status = h.async_pwrite(aio_buffer, aio_file, 0)
        assert write_status == 0

        wait_status = h.wait()
        assert wait_status == 1

        if use_cpu_locked_tensor:
            h.free_cpu_locked_tensor(aio_buffer)

        assert os.path.isfile(aio_file)

        filecmp.clear_cache()
        assert filecmp.cmp(ref_file, aio_file, shallow=False)


# @pytest.mark.sequential
@pytest.mark.parametrize("use_cuda_pinned_tensor", [True, False])
@pytest.mark.parametrize("use_unpinned_tensor", [True, False])
class TestAsyncQueue(DistributedTest):
    world_size = 1
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize("async_queue", [2, 3])
    def test_read(self, tmpdir, async_queue, use_cuda_pinned_tensor, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        block_size, io_size = _get_block_and_io_size(tmpdir)

        ref_files = []
        for i in range(async_queue):
            f, _ = _do_ref_write(tmpdir, io_size, i)
            ref_files.append(f)

        single_submit = True
        overlap_events = True
        h = AsyncIOBuilder().load().aio_handle(block_size, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        use_cpu_locked_tensor = False
        if use_unpinned_tensor:
            aio_buffers = [
                torch.empty(io_size, dtype=torch.uint8, device=get_accelerator().device_name())
                for _ in range(async_queue)
            ]
        elif use_cuda_pinned_tensor:
            aio_buffers = [
                get_accelerator().pin_memory(torch.empty(io_size, dtype=torch.uint8, device='cpu'))
                for _ in range(async_queue)
            ]
        else:
            tmp_tensor = torch.empty(0, dtype=torch.uint8)
            aio_buffers = [h.new_cpu_locked_tensor(io_size, tmp_tensor) for _ in range(async_queue)]
            use_cpu_locked_tensor = True

        _validate_handle_state(h, single_submit, overlap_events, block_size)

        for i in range(async_queue):
            read_status = h.async_pread(aio_buffers[i], ref_files[i], 0)
            assert read_status == 0

        wait_status = h.wait()
        assert wait_status == async_queue

        for i in range(async_queue):
            with open(ref_files[i], 'rb') as f:
                ref_buffer = list(f.read())
            assert ref_buffer == aio_buffers[i].tolist()

        if use_cpu_locked_tensor:
            for t in aio_buffers:
                h.free_cpu_locked_tensor(t)

    @pytest.mark.parametrize("async_queue", [2, 3])
    def test_write(self, tmpdir, use_cuda_pinned_tensor, async_queue, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        block_size, io_size = _get_block_and_io_size(tmpdir)

        ref_files = []
        ref_buffers = []
        for i in range(async_queue):
            f, buf = _do_ref_write(tmpdir, io_size, i)
            ref_files.append(f)
            ref_buffers.append(buf)

        single_submit = True
        overlap_events = True
        h = AsyncIOBuilder().load().aio_handle(block_size, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        aio_files = []
        aio_buffers = []
        for i in range(async_queue):
            if use_unpinned_tensor:
                f, buf = _get_test_write_file_and_unpinned_tensor(tmpdir, ref_buffers[i], i)
            elif use_cuda_pinned_tensor:
                f, buf = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffers[i], None, i)
            else:
                f, buf = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffers[i], h, i)
            aio_files.append(f)
            aio_buffers.append(buf)

        use_cpu_locked_tensor = not (use_unpinned_tensor or use_cuda_pinned_tensor)

        _validate_handle_state(h, single_submit, overlap_events, block_size)

        for i in range(async_queue):
            read_status = h.async_pwrite(aio_buffers[i], aio_files[i], 0)
            assert read_status == 0

        wait_status = h.wait()
        assert wait_status == async_queue

        if use_cpu_locked_tensor:
            for t in aio_buffers:
                h.free_cpu_locked_tensor(t)

        for i in range(async_queue):
            assert os.path.isfile(aio_files[i])

            filecmp.clear_cache()
            assert filecmp.cmp(ref_files[i], aio_files[i], shallow=False)


@pytest.mark.parametrize("use_cuda_pinned_tensor", [True, False])
@pytest.mark.parametrize('file_partitions', [[1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1]])
class TestAsyncFileOffset(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize('use_fd', [False, True])
    def test_offset_write(self, tmpdir, file_partitions, use_cuda_pinned_tensor, use_fd):

        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        ref_file = _get_file_path(tmpdir, '_py_random')
        aio_file = _get_file_path(tmpdir, '_aio_random')
        block_size, _ = _get_block_and_io_size(tmpdir)
        # Each partition is split across IO_PARALLEL threads, so size it to keep every thread chunk aligned.
        partition_unit_size = block_size * IO_PARALLEL
        file_size = sum(file_partitions) * partition_unit_size

        h = AsyncIOBuilder().load().aio_handle(block_size, QUEUE_DEPTH, True, True, IO_PARALLEL)

        if use_cuda_pinned_tensor:
            data_buffer = torch.ByteTensor(list(os.urandom(file_size))).pin_memory()
        else:
            data_buffer = h.new_cpu_locked_tensor(file_size, torch.empty(0, dtype=torch.uint8))

        file_offsets = []
        next_offset = 0
        for i in range(len(file_partitions)):
            file_offsets.append(next_offset)
            next_offset += file_partitions[i] * partition_unit_size

        ref_fd = open(ref_file, 'wb')
        for i in range(len(file_partitions)):
            src_buffer = torch.narrow(data_buffer, 0, file_offsets[i], file_partitions[i] * partition_unit_size)

            ref_fd.write(src_buffer.numpy().tobytes())
            ref_fd.flush()

            if use_fd:
                aio_fd = os.open(aio_file, flags=os.O_DIRECT | os.O_CREAT | os.O_WRONLY)
                write_status = h.async_pwrite(buffer=src_buffer, fd=aio_fd, file_offset=file_offsets[i])
            else:
                write_status = h.async_pwrite(buffer=src_buffer, filename=aio_file, file_offset=file_offsets[i])
            assert write_status == 0
            wait_status = h.wait()
            assert wait_status == 1

            if use_fd:
                os.path.isfile(aio_fd)
                os.close(aio_fd)

            filecmp.clear_cache()
            assert filecmp.cmp(ref_file, aio_file, shallow=False)

        ref_fd.close()

        if not use_cuda_pinned_tensor:
            h.free_cpu_locked_tensor(data_buffer)

    def _test_offset_read(self, tmpdir, file_partitions, use_cuda_pinned_tensor):

        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        block_size, _ = _get_block_and_io_size(tmpdir)
        # Each partition is split across IO_PARALLEL threads, so size it to keep every thread chunk aligned.
        partition_unit_size = block_size * IO_PARALLEL
        file_size = sum(file_partitions) * partition_unit_size
        ref_file, _ = _do_ref_write(tmpdir, file_size)
        h = AsyncIOBuilder().load().aio_handle(block_size, QUEUE_DEPTH, True, True, IO_PARALLEL)

        if use_cuda_pinned_tensor:
            data_buffer = torch.zeros(file_size, dtype=torch.uint8, device='cpu').pin_memory()
        else:
            data_buffer = h.new_cpu_locked_tensor(file_size, torch.empty(0, dtype=torch.uint8))

        file_offsets = []
        next_offset = 0
        for i in range(len(file_partitions)):
            file_offsets.append(next_offset)
            next_offset += file_partitions[i] * partition_unit_size

        with open(ref_file, 'rb') as ref_fd:
            for i in range(len(file_partitions)):
                ref_fd.seek(file_offsets[i])
                bytes_to_read = file_partitions[i] * partition_unit_size
                ref_buf = list(ref_fd.read(bytes_to_read))

                dst_tensor = torch.narrow(data_buffer, 0, 0, bytes_to_read)
                assert 1 == h.sync_pread(dst_tensor, ref_file, file_offsets[i])
                assert dst_tensor.tolist() == ref_buf

        if not use_cuda_pinned_tensor:
            h.free_cpu_locked_tensor(data_buffer)


class TestAioAlignedBytes:

    def test_probe_matches_filesystem(self, tmpdir):
        get_aio_aligned_bytes.cache_clear()
        aligned_bytes = get_aio_aligned_bytes(str(tmpdir))
        assert aligned_bytes in (1024, 2048, 4096)
        assert os.listdir(str(tmpdir)) == []

    def test_probe_on_4k_native_device(self, tmpdir, monkeypatch):
        # Emulate a 4Kn device, where O_DIRECT rejects any I/O not aligned to 4096 bytes.
        real_pwritev = os.pwritev

        def pwritev_4k_only(fd, buffers, offset):
            num_bytes = sum(len(b) for b in buffers)
            if num_bytes % 4096 or offset % 4096:
                raise OSError(22, "Invalid argument")
            return real_pwritev(fd, buffers, offset)

        monkeypatch.setattr(os, "pwritev", pwritev_4k_only)
        get_aio_aligned_bytes.cache_clear()
        assert get_aio_aligned_bytes(str(tmpdir)) == 4096
        get_aio_aligned_bytes.cache_clear()

    def test_probe_without_odirect_support(self, tmpdir, monkeypatch):
        real_open = os.open

        def open_without_odirect(path, flags, *args):
            if flags & os.O_DIRECT:
                raise OSError(22, "Invalid argument")
            return real_open(path, flags, *args)

        monkeypatch.setattr(os, "open", open_without_odirect)
        get_aio_aligned_bytes.cache_clear()
        assert get_aio_aligned_bytes(str(tmpdir)) == AIO_ALIGNED_BYTES
        get_aio_aligned_bytes.cache_clear()
