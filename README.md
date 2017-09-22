#  TL;DR

Following this instructions is probably enough to use TensorBoard to visualize data from Python. To develop plugins, read further.

1. clone this repo
1. `pip install tensorflow-tensorboard` # this is the TB standalone (no needed if TF already installed)
1. `pip install tensorboard-pytorch` # this is the actual logger
1. run `python demo_builtin_visualizations.py` # writes some data into the `./logs` dir
1. run `tensorboard --logdir=logs` # tell the TB server to send the data in the `./logs` dir via `0.0.0.0:6006`
1. browse `localhost:6006` # visualize the data served by TB with any web browser (chrome works, others may not).


# TensorBoard Mechanics

TensorBoard (TB) is a suite of [machine-learning related visualization tools](https://raw.githubusercontent.com/lanpa/tensorboard-pytorch/master/screenshots/Demo.gif) designed to run on a HTTP server, so they can be then visited by a web browser locally and remotely. It is delivered as a part of the TensorFlow (TF) libraries, and the [TF tutorials](https://www.tensorflow.org/api_guides/python/summary) provide an explanation for the TF-specific usage, but it can be used with other platforms like PyTorch  (PT) or simply from Python as explained in [this SO post](https://stackoverflow.com/questions/37902705/how-to-manually-create-a-tf-summary) and described later (Actually, it can be run from other languages like [Ruby](https://github.com/somaticio/tensorflow.rb) as well but only Python will be covered here).

As for September 2017, TensorBoard provides built-in visualization support for **scalar functions, images, histograms, text, audio and distributions**, as well as for **projector and profiling data**. For the TF and PT platforms, it also provides an inspection tool for the **computational graph**. also, via its [recently released api](https://github.com/tensorflow/tensorboard-plugin-example), it is possible to develope **plugins** to provide support for custom visualizations and UI. this is also covered here.


### Logdir and Summaries

once started, the tb server inspects periodically a **log directory** (logdir) that holds the data generated by the python code. if new information is found, the visualizations are updated. the logdir has to be specified when starting the tb server as follows: `tensorboard --logdir=<log_directory>`. the ip and port can also be specified (the default value is `0.0.0.0:6006`).

this logdir contains a set of [protocol buffers](https://developers.google.com/protocol-buffers/) (protobufs), which are basically extensible strings that contain data like functions, images, audio files, as well as the metadata attached to them like date and title. especially important is the `tag` field, which is a string that is uniquely associated with a certain “stream” of related data: when writing many sumaries with the same `tag`, they will be visualized as belonging to the same group (for instance, the same function for scalars,or the same image cluster for images).

as described in [tf tutorials](https://www.tensorflow.org/api_guides/python/summary), the `tf.summary.filewriter` python class manages the output to the protobufs, and the writing operations are performed by functions like `tf.summary.tensor_summary`, `tf.summary.scalar`, `tf.summary.histogram`, `tf.summary.audio` or `tf.summary.image`, depending on the desired visualization.

the contents of the writing operation are represented in tensorflow by objects of the [`tf.summary`](https://www.tensorflow.org/api_docs/python/tf/summary) class, which can accept numpy arrays or other types of python's numerical collections or strings as their construction arguments. when passed to the writer together with a tag, they get conveniently translated to the corresponding protobuf.

this allows us to define a logger class that encapsulates all this functionality, as explained in the next subsection.


### The Logger Class

There are already several projects already that try to provide a clean and flexible interface between any python program and the tb server:

* [tensorboard-logger](https://github.com/TeamHG-Memex/tensorboard_logger): it doesn't require to import TF at all, but works only for scalars
* [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard): it allows logging of scalars, images and histograms only. It is specific for PT
* [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) includes support for TB's newest functionality, except of plugins. It is also PT-specific

Among them, the package [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) (installable via `pip`) is the most complete one, since it implements interface for all TB's built-in visualizations. A brief analysis of its contents can be found [here](lanpa_description.md). The idea of the present repo is to document how to use it (or extend it without losing any of its functionality), for the two following goals:

* Allow its logger functions to directly process numpy and python native datatypes (that is, not being PT-specific)
* include all of TB's functionality, including the recently released plugin API.

For the first goal, take a look at the [`demo_builtin_visualizations.py`](demo_builtin_visualizations.py) file, which has an example for every one of the built-in visualizations. For the second one, read further.


### TensorBoard Plugin API

Apart from the built-in support for the visualization of scalars, images and such, TB has released an API that allows developers to create new, custom visualizations. For that, it is necessary to complete the description of TB's data model:  we already explained that the information unit is the `Summary`, and that its `tag` attribute relates different summaries to the same “data stream”. The other three needed concepts are:

* The [`SummaryMetadata`](https://github.com/tensorflow/tensorflow/blob/e9d5ee1ebffba25cef65f1f354b9e4ca9bcea10c/tensorflow/core/framework/summary.proto#L38) object, contains metadata associated with the plugin.
* The [`TensorProto`](https://github.com/tensorflow/tensorflow/blob/e9d5ee1ebffba25cef65f1f354b9e4ca9bcea10c/tensorflow/core/framework/tensor.proto#L14) is a very general container, you can put any data here so it is useful for building arbitrary plugins.
* The `run`: this is right between the idea of `tag`, which refers to many summaries of the same stream, and the idea of `logdir`, which contains all the data displayed by the TB server. Runs are folders in the logdir that contain many datastreams.

So summarizing, when running a Python script, and desiring to visualize the evolution of different variables: every single recorded value goes to a `Summary`, the summaries of a specific variable belong to the same `tag`, the whole data recorded by the script belongs to the same `run`, and the programmer of the plugin has to take care of the `SummaryMetadata` and to put the info into `TensorProto` containers (see [this readme](https://github.com/tensorflow/tensorboard-plugin-example) for more details).

To develop the plugins, there is the Python part (create a `summary` operation, and its implementation NOTE THAT THIS IS UNDER ACTIVE DEVELOPMENT SEE [HERE](https://github.com/tensorflow/tensorboard-plugin-example)) for the backend+API and the JavaScript part for the frontend.


### SSH TUnneling

In some cases, the computer holding the log files is only accessible via SSH and prevents visiting the TensorBoard HTTP server. In that case, SSH tunneling can be performed: The following command forwards TensorBoard to `localhost:16006`, assuming it was started with the default IP address:  

`ssh -p <PORT> <USER>@<SERVER> -N -f -L localhost:16006:localhost:6006`

## TODO
* expand the pip logger with [greeter](https://github.com/tensorflow/tensorboard-plugin-example) plugin. then for the [beholder](https://github.com/chrisranderson/beholder) plugin.
* finally expand functionality with further plugins (3D t-sne, videos, TB SNAPSHOT)
* document it well (that is, finish the TensorBoard Plugin API and the lanpa_description.md)

