import asyncio
from os import getenv
from bullmq import Worker
from recognize import Gender, Recognizer

recognizer = Recognizer()


async def is_male(job, *args) -> int:
    file_path = job.data["filePath"]
    print(f'Recognizing {file_path}...')
    result = recognizer.recognize(file_path)
    print(f'Done. Result: {result}')
    return result == Gender.Male


if __name__ == '__main__':
    worker = Worker("is-male-voice", is_male, {'autorun': False, 'connection': {
        'host': getenv('REDIS_QUEUE_HOST'),
    }})

    loop = asyncio.get_event_loop()
    loop.run_until_complete(worker.run())
