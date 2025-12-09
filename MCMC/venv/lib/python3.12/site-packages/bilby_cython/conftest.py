import pytest
import numpy as np


@pytest.fixture(params=[np.int64, np.int32, np.int16, np.float64, np.float32])
def types(request):
    return request.param
