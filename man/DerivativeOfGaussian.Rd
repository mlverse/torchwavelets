% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wavelets.R
\name{DerivativeOfGaussian}
\alias{DerivativeOfGaussian}
\title{Derivative of Gaussian of order `m`}
\description{
Derivative of Gaussian of order `m`

Derivative of Gaussian of order `m`
}
\details{
When m = 2, this is also known as the "Mexican hat", "Marr" or "Ricker" wavelet.
It models the function:
``A d^m/dx^m exp(-x^2 / 2)``,
where ``A = (-1)^(m+1) / (gamma(m + 1/2))^.5`` and   ``x = t / s``.
Note that the energy of the return wavelet is not normalized according to `s`.
}
\note{
The derivative of the Gaussian has a polynomial representation.
From http://en.wikipedia.org/wiki/Gaussian_function:
"Mathematically, the derivatives of the Gaussian function can be represented
using Hermite functions. The n-th derivative of the Gaussian is the Gaussian
function itself multiplied by the n-th Hermite polynomial, up to scale."
For Hermite polynomials, see http://en.wikipedia.org/wiki/Hermite_polynomial.
Here, we want the "probabilists" Hermite polynomial (He_n),
which may be computed using `calculus::hermite`.
}
\section{Super class}{
\code{\link[torchwavelets:Wavelet]{torchwavelets::Wavelet}} -> \code{Derivative of Gaussian}
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{is_complex}}{whether the wavelet representation in the time domain is complex}

\item{\code{m}}{the order of the derivative}

\item{\code{C_d}}{value of C_d from TC98}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-Derivative of Gaussian-new}{\code{DerivativeOfGaussian$new()}}
\item \href{#method-Derivative of Gaussian-time}{\code{DerivativeOfGaussian$time()}}
\item \href{#method-Derivative of Gaussian-fourier_period}{\code{DerivativeOfGaussian$fourier_period()}}
\item \href{#method-Derivative of Gaussian-scale_from_period}{\code{DerivativeOfGaussian$scale_from_period()}}
\item \href{#method-Derivative of Gaussian-frequency}{\code{DerivativeOfGaussian$frequency()}}
\item \href{#method-Derivative of Gaussian-coi}{\code{DerivativeOfGaussian$coi()}}
\item \href{#method-Derivative of Gaussian-clone}{\code{DerivativeOfGaussian$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Derivative of Gaussian-new"></a>}}
\if{latex}{\out{\hypertarget{method-Derivative of Gaussian-new}{}}}
\subsection{Method \code{new()}}{
save `m` and set `C_d`
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DerivativeOfGaussian$new(m = 2)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{m}}{the order of the derivative}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Derivative of Gaussian-time"></a>}}
\if{latex}{\out{\hypertarget{method-Derivative of Gaussian-time}{}}}
\subsection{Method \code{time()}}{
value of the wavelet at the given times
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DerivativeOfGaussian$time(t, s = 1)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{t}}{time. If `s` is not specified, this can be used as the
non-dimensional time t/s.}

\item{\code{s}}{scaling factor. Default is 1.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Derivative of Gaussian-fourier_period"></a>}}
\if{latex}{\out{\hypertarget{method-Derivative of Gaussian-fourier_period}{}}}
\subsection{Method \code{fourier_period()}}{
equivalent Fourier period
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DerivativeOfGaussian$fourier_period(s)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{s}}{scaling factor}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Derivative of Gaussian-scale_from_period"></a>}}
\if{latex}{\out{\hypertarget{method-Derivative of Gaussian-scale_from_period}{}}}
\subsection{Method \code{scale_from_period()}}{
compute the scale from the fourier period
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DerivativeOfGaussian$scale_from_period(period)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{period}}{Fourier period}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Derivative of Gaussian-frequency"></a>}}
\if{latex}{\out{\hypertarget{method-Derivative of Gaussian-frequency}{}}}
\subsection{Method \code{frequency()}}{
frequency representation of wavelet
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DerivativeOfGaussian$frequency(w, s = 1)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{w}}{angular frequency. If `s` is not specified, i.e. set to 1,
this can be used as the non-dimensional angular frequency w * s.}

\item{\code{s}}{the scaling factor. Default is 1.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Derivative of Gaussian-coi"></a>}}
\if{latex}{\out{\hypertarget{method-Derivative of Gaussian-coi}{}}}
\subsection{Method \code{coi()}}{
The e-folding time for the autocorrelation of wavelet
power at each scale, i.e. the timescale over which an edge
effect decays by a factor of 1/e^2.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DerivativeOfGaussian$coi(s)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{s}}{scaling factor}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Derivative of Gaussian-clone"></a>}}
\if{latex}{\out{\hypertarget{method-Derivative of Gaussian-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DerivativeOfGaussian$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
