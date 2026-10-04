import asyncio

import pytest
from fastapi import HTTPException
from gateway.admission import AdmissionQueue, AdmittedStreamingResponse, until_disconnect, QueueFull


async def queued(q, count):
    async with asyncio.timeout(1):
        while q.stats()['queued'] != count:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_five_slots_and_fifo():
    q = AdmissionQueue(5, 10, 1)
    leases = [await q.acquire('qwen') for _ in range(5)]
    sixth = asyncio.create_task(q.acquire('qwen'))
    seventh = asyncio.create_task(q.acquire('qwen'))
    await queued(q, 2)
    assert q.active == 5
    leases[0].release()
    six = await sixth
    assert not seventh.done()
    assert q.active == 5
    leases[1].release()
    seven = await seventh
    for lease in [*leases, six, seven]:
        lease.release()
    assert q.active == 0


@pytest.mark.asyncio
async def test_profile_change_drains_and_cannot_be_starved():
    q = AdmissionQueue(5, 10, 1)
    first = await q.acquire('qwen')
    other = asyncio.create_task(q.acquire('deepseek'))
    late = asyncio.create_task(q.acquire('qwen'))
    await queued(q, 2)
    assert q.active == 1
    first.release()
    second = await other
    assert not late.done()
    second.release()
    (await late).release()
    assert q.active == 0


@pytest.mark.asyncio
async def test_full_timeout_and_cancel_remove_waiters():
    q = AdmissionQueue(1, 1, .03)
    lease = await q.acquire('qwen')
    waiting = asyncio.create_task(q.acquire('qwen'))
    await queued(q, 1)
    with pytest.raises(QueueFull):
        await q.acquire('qwen')
    with pytest.raises(asyncio.TimeoutError):
        await waiting
    assert q.stats()['queued'] == 0
    waiting = asyncio.create_task(q.acquire('qwen'))
    await queued(q, 1)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert q.stats()['queued'] == 0
    lease.release()
    assert q.active == 0


@pytest.mark.asyncio
async def test_cancel_racing_with_grant_does_not_leak():
    q = AdmissionQueue(1, 1, 1)
    first = await q.acquire('qwen')
    waiting = asyncio.create_task(q.acquire('qwen'))
    await queued(q, 1)
    first.release()
    waiting.cancel()
    try:
        lease = await waiting
    except asyncio.CancelledError:
        pass
    else:
        lease.release()
    assert q.active == 0
    assert q.stats()['queued'] == 0


@pytest.mark.asyncio
async def test_disconnect_cancels_waiting_operation():
    messages = asyncio.Queue()
    class Request:
        receive = messages.get
    q = AdmissionQueue(1, 2, 1)
    first = await q.acquire('qwen')
    task = asyncio.create_task(until_disconnect(Request(), lambda: q.acquire('qwen')))
    await queued(q, 1)
    await messages.put({'type': 'http.disconnect'})
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 499
    assert q.stats()['queued'] == 0
    first.release()


@pytest.mark.asyncio
@pytest.mark.parametrize('disconnect', [False, True])
async def test_stream_holds_slot_and_closes_backend(disconnect):
    q = AdmissionQueue(1, 2, 1)
    lease = await q.acquire('qwen')
    sent = asyncio.Event()
    finish = asyncio.Event()
    closed = asyncio.Event()
    messages = asyncio.Queue()
    async def body():
        try:
            yield b'data: first\n\n'
            await finish.wait()
            yield b'data: [DONE]\n\n'
        finally:
            closed.set()
    async def send(message):
        if message['type'] == 'http.response.body':
            sent.set()
    response = AdmittedStreamingResponse(body())
    response.lease = lease
    task = asyncio.create_task(response({'type': 'http', 'asgi': {'spec_version': '2.3'}}, messages.get, send))
    await asyncio.wait_for(sent.wait(), 1)
    waiting = asyncio.create_task(q.acquire('qwen'))
    await queued(q, 1)
    assert not waiting.done()
    if disconnect:
        await messages.put({'type': 'http.disconnect'})
    else:
        finish.set()
    await asyncio.wait_for(task, 1)
    assert closed.is_set()
    (await waiting).release()
    assert q.active == 0


@pytest.mark.asyncio
async def test_disconnect_cancels_active_backend():
    messages = asyncio.Queue()
    class Request:
        receive = messages.get
    started = asyncio.Event()
    stopped = asyncio.Event()
    async def operation():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            stopped.set()
    task = asyncio.create_task(until_disconnect(Request(), operation))
    await started.wait()
    await messages.put({'type': 'http.disconnect'})
    with pytest.raises(HTTPException):
        await task
    assert stopped.is_set()
