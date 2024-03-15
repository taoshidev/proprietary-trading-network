# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import os


class ValiMemoryUtils:

    @staticmethod
    def get_vali_memory() -> str:
        return os.getenv("vm")

    @staticmethod
    def set_vali_memory(vm) -> None:
        os.environ["vm"] = vm
