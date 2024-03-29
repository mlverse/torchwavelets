% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wavelets.R
\name{Wavelet}
\alias{Wavelet}
\title{Abstract wavelet base class, guaranteeing all required methods are implemented.}
\description{
Abstract wavelet base class, guaranteeing all required methods are implemented.

Abstract wavelet base class, guaranteeing all required methods are implemented.
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-Wavelet-new}{\code{Wavelet$new()}}
\item \href{#method-Wavelet-time}{\code{Wavelet$time()}}
\item \href{#method-Wavelet-fourier_period}{\code{Wavelet$fourier_period()}}
\item \href{#method-Wavelet-scale_from_period}{\code{Wavelet$scale_from_period()}}
\item \href{#method-Wavelet-frequency}{\code{Wavelet$frequency()}}
\item \href{#method-Wavelet-coi}{\code{Wavelet$coi()}}
\item \href{#method-Wavelet-clone}{\code{Wavelet$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Wavelet-new"></a>}}
\if{latex}{\out{\hypertarget{method-Wavelet-new}{}}}
\subsection{Method \code{new()}}{
use this to set any wavelet-specific parameters
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Wavelet$new()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Wavelet-time"></a>}}
\if{latex}{\out{\hypertarget{method-Wavelet-time}{}}}
\subsection{Method \code{time()}}{
value of the wavelet at the given times
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Wavelet$time(t, s = 1, ...)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{t}}{time. If `s` is not specified, this can be used as the non-dimensional
time t/s.}

\item{\code{s}}{scaling factor. Default is 1.}

\item{\code{...}}{additional wavelet-specific parameters}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Wavelet-fourier_period"></a>}}
\if{latex}{\out{\hypertarget{method-Wavelet-fourier_period}{}}}
\subsection{Method \code{fourier_period()}}{
equivalent Fourier period
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Wavelet$fourier_period(s)}\if{html}{\out{</div>}}
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
\if{html}{\out{<a id="method-Wavelet-scale_from_period"></a>}}
\if{latex}{\out{\hypertarget{method-Wavelet-scale_from_period}{}}}
\subsection{Method \code{scale_from_period()}}{
compute the scale from the fourier period
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Wavelet$scale_from_period(p)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{p}}{Fourier period}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Wavelet-frequency"></a>}}
\if{latex}{\out{\hypertarget{method-Wavelet-frequency}{}}}
\subsection{Method \code{frequency()}}{
frequency representation
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Wavelet$frequency(w, s = 1)}\if{html}{\out{</div>}}
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
\if{html}{\out{<a id="method-Wavelet-coi"></a>}}
\if{latex}{\out{\hypertarget{method-Wavelet-coi}{}}}
\subsection{Method \code{coi()}}{
The e-folding time for the autocorrelation of wavelet
power at each scale, i.e. the timescale over which an edge
effect decays by a factor of 1/e^2.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Wavelet$coi(s)}\if{html}{\out{</div>}}
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
\if{html}{\out{<a id="method-Wavelet-clone"></a>}}
\if{latex}{\out{\hypertarget{method-Wavelet-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Wavelet$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
