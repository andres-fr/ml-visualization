
### About TensorBoard

TensorBoard is a suite of visualization tools designed to run on a HTTP server, that can be then visited by a web browser. It provides real-time information on the implemented model, by showing its computational graph as well as the functions, images, audio files and/or histograms that it may generate. For that, the server visits a log directory that has to be specified when it starts with the `--logdir` flag, and serves to a given IP address (`localhost:6006` by default).

This log directory contains a set of protocol buffers\cite{protobufs}, which are basically extensible strings that contain data like functions, images, audio files, as well as the metadata attached to them like date and title. This buffers are generated and interactively expanded by the `tf.summary.FileWriter`, and the contents are represented in TensorFlow by the `tf.Summary` objects.

The `TensorBoardLogger` class in the `3\_main\_pipeline.py` Python script provides an implementation that outputs functions and matrix plots to the specified log directory.

In some cases, the computer holding the log files is only accessible via SSH and prevents visiting the TensorBoard HTTP server. In that case, SSH tunneling can be performed: The following command forwards TensorBoard to `localhost:16006`, assuming it was started with the default IP address:
`ssh -p <PORT> <USER>@<SERVER> -N -f -L localhost:16006:localhost:6006`
https://git.ccc.cs.uni-frankfurt.de/rodriguez/dnc-tracking/blob/master/wiki/basic_interactions.md
