# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

import paddle
from paddle.distribution.normal import Normal
from paddle.distribution.transform import ExpTransform
from paddle.distribution.transformed_distribution import TransformedDistribution

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from typing_extensions import TypeAlias

    from paddle import Tensor
    from paddle._typing import NestedSequence

    _LognormalLocBase: TypeAlias = Union[float, complex]
    _LognormalLocNDArray: TypeAlias = Union[
        np.float32, np.float64, np.complex64, np.complex128
    ]
    _LognormalLoc: TypeAlias = Union[
        _LognormalLocBase,
        Sequence[_LognormalLocBase],
        NestedSequence[_LognormalLocBase],
        npt.NDArray[_LognormalLocNDArray],
        Tensor,
    ]
    _LognormalScale: TypeAlias = Union[
        float,
        Sequence[float],
        NestedSequence[float],
        npt.NDArray[Union[np.float32, np.float64]],
        Tensor,
    ]


class LogNormal(TransformedDistribution):
    r"""The LogNormal distribution with location `loc` and `scale` parameters.

    .. math::

        X \sim Normal(\mu, \sigma)

        Y = exp(X) \sim LogNormal(\mu, \sigma)


    Due to LogNormal distribution is based on the transformation of Normal distribution, we call that :math:`Normal(\mu, \sigma)` is the underlying distribution of :math:`LogNormal(\mu, \sigma)`

    Mathematical details

    The probability density function (pdf) is

    .. math::
        pdf(x; \mu, \sigma) = \frac{1}{\sigma x \sqrt{2\pi}}e^{(-\frac{(ln(x) - \mu)^2}{2\sigma^2})}

    In the above equation:

    * :math:`loc = \mu`: is the means of the underlying Normal distribution.
    * :math:`scale = \sigma`: is the stddevs of the underlying Normal distribution.

    Args:
        loc(int|float|complex|list|tuple|numpy.ndarray|Tensor): The means of the underlying Normal distribution.The data type is float32, float64, complex64 and complex128.
        scale(int|float|list|tuple|numpy.ndarray|Tensor): The stddevs of the underlying Normal distribution.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import LogNormal

            >>> # Define a single scalar LogNormal distribution.
            >>> dist = LogNormal(loc=0., scale=3.)
            >>> # Define a batch of two scalar valued LogNormals.
            >>> # The underlying Normal of first has mean 1 and standard deviation 11, the underlying Normal of second 2 and 22.
            >>> dist = LogNormal(loc=[1., 2.], scale=[11., 22.])
            >>> # Get 3 samples, returning a 3 x 2 tensor.
            >>> dist.sample((3, ))

            >>> # Define a batch of two scalar valued LogNormals.
            >>> # Their underlying Normal have mean 1, but different standard deviations.
            >>> dist = LogNormal(loc=1., scale=[11., 22.])

            >>> # Complete example
            >>> value_tensor = paddle.to_tensor([0.8], dtype="float32")

            >>> lognormal_a = LogNormal([0.], [1.])
            >>> lognormal_b = LogNormal([0.5], [2.])
            >>> sample = lognormal_a.sample((2, ))
            >>> # a random tensor created by lognormal distribution with shape: [2, 1]
            >>> entropy = lognormal_a.entropy()
            >>> print(entropy)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [1.41893852])
            >>> lp = lognormal_a.log_prob(value_tensor)
            >>> print(lp)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [-0.72069150])
            >>> p = lognormal_a.probs(value_tensor)
            >>> print(p)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.48641577])
            >>> kl = lognormal_a.kl_divergence(lognormal_b)
            >>> print(kl)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.34939718])
    """
    loc: Tensor
    scale: Tensor

    def __init__(self, loc: _LognormalLoc, scale: _LognormalScale) -> None:
        self._base = Normal(loc=loc, scale=scale)
        self.loc = self._base.loc
        self.scale = self._base.scale
        super().__init__(self._base, [ExpTransform()])

    @property
    def mean(self) -> Tensor:
        """Mean of lognormal distribution.

        Returns:
            Tensor: mean value.
        """
        return paddle.exp(self._base.mean + self._base.variance / 2)

    @property
    def variance(self) -> Tensor:
        """Variance of lognormal distribution.

        Returns:
            Tensor: variance value.
        """
        return paddle.expm1(self._base.variance) * paddle.exp(
            2 * self._base.mean + self._base.variance
        )

    def entropy(self) -> Tensor:
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            entropy(\sigma) = 0.5 \log (2 \pi e \sigma^2) + \mu

        In the above equation:

        * :math:`loc = \mu`: is the mean of the underlying Normal distribution.
        * :math:`scale = \sigma`: is the stddevs of the underlying Normal distribution.

        Returns:
          Tensor: Shannon entropy of lognormal distribution.

        """
        return self._base.entropy() + self._base.mean

    def probs(self, value: Tensor) -> Tensor:
        """Probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: probability.The data type is same with :attr:`value` .

        """
        return paddle.exp(self.log_prob(value))

    def kl_divergence(self, other: LogNormal) -> Tensor:
        r"""The KL-divergence between two lognormal distributions.

        The probability density function (pdf) is

        .. math::

            KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio})

        .. math::

            ratio = \frac{\sigma_0}{\sigma_1}

        .. math::

            diff = \mu_1 - \mu_0

        In the above equation:

        * :math:`loc = \mu_0`: is the means of current underlying Normal distribution.
        * :math:`scale = \sigma_0`: is the stddevs of current underlying Normal distribution.
        * :math:`loc = \mu_1`: is the means of other underlying Normal distribution.
        * :math:`scale = \sigma_1`: is the stddevs of other underlying Normal distribution.
        * :math:`ratio`: is the ratio of scales.
        * :math:`diff`: is the difference between means.

        Args:
            other (LogNormal): instance of LogNormal.

        Returns:
            Tensor: kl-divergence between two lognormal distributions.

        """
        return self._base.kl_divergence(other._base)
