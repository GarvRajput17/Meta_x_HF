# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CivicFlow server-side components.

The environment class lives in `server.civicflow_env_environment`. We
intentionally do NOT eagerly import it here, because doing so creates a
cycle: tasks.py imports server.state, which would otherwise force the
environment module to load mid-package-init. Import the environment
directly when you need it.
"""
