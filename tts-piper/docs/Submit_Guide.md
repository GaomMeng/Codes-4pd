# Contest Submit Guide
This guide aims to help you quickly build your own docker image, and submit to [http://contest.4pd.io](http://contest.4pd.io).

## Level 0 - Resubmit A Previous Submit
* The following is a sample submit to the Greek TTS task
```yaml
docker_image: harbor.4pd.io/lab-platform/pk_platform/model_services/liuxinyang/piper-tts:0.0.1-b

values:
  env:
    - name: TTS_MODEL_PATH
      value: "/mnt/models/tts/el/el_GR-rapunzelina-low.onnx"
  
  resources:
    limits:
      cpu: 8
      memory: 32Gi
    requests:
      cpu: 8
      memory: 32Gi

  readinessProbe:
    httpGet:
      path: /ready
      port: 80 
    initialDelaySeconds: 30
    periodSeconds: 10 
    timeoutSeconds: 1

leaderboard_options:
  nfs:
    - name: data_path
      srcRelativePath: liuxinyang/models/tts/el
      mountPoint: /mnt/models/tts/el
      source: ceph_customer
```

## Level 1 - Submit with Some Other Models on NFS

1. Download your models from [piper-voices](https://huggingface.co/rhasspy/piper-voices) together with the json config file.

2. Upload the model to `ceph_customer` or other NFS storage. You can refer to [this document](https://wiki.4paradigm.com/pages/viewpage.action?pageId=140889286#id-算法竞赛场使用FAQ-Q1.1如何加载我自己的大模型、词表、数据等镜像运行需要的大文件) on how to manage the NFS.

3. Edit the submit info.
    1. Change `TTS_MODEL_PATH`
        ```yaml
        values:
          env:
            - name: TTS_MODEL_PATH
              value: "the/mounted/path/to/your/onnx/model/file"
        ```
    2. Change NFS config
        ```yaml
        leaderboard_options:
          nfs:
            - name: data_path
              srcRelativePath: relative/path/in/nfs
              mountPoint: absolute/path/to/mount/to/container
              source: ceph_customer/ucloud_juicefs
        ```
    3. Submit and debug

## Level 2 - Create Your Own Docker Image
1. Clone the repository and read the code.
2. Make changes.
3. Rebuild your own docker, and push to the docker harbor.
    * For faster building, you may refer to the following Dockerfile instead of the original one.
    ```Dockerfile
    # This is a module with all the dependencies required for TTS.
    FROM harbor.4pd.io/lab-platform/pk_platform/model_services/liuxinyang/piper-tts-base:0.0.0

    # If you import new pip modules, you will need to update requirements.txt
    # COPY requirements .
    # Install dependencies
    #
    # RUN pip install -r requirements.txt 
    # 
    # If you have network problems and need to use other mirrors,
    # For example you are packing the docker on a server of the company without
    # Internet access, you can use the following command.
    #
    # RUN pip install -r requirements.txt -i https://nexus.4pd.io/repository/pypi-all/simple
    #
    # Which utilizes the company's mirror of PyPI. 
    #
    # *You may fail to find the module you need.*

    # Copy the modified code and replace the original in the base image.
    COPY tts tts

    # Copy other necessary files
    # For example, if you decide to put the model file INSIDE the image
    # COPY piper_tts_model.onnx /mnt/models/
    # COPY piper_tts_model.onnx.json /mnt/models/
    # Remember to adjust the TTS_MODEL_PATH in your submit.

    # Command to run your application
    CMD ["uvicorn", "tts.server:app", "--host", "0.0.0.0", "--port", "80"]

    ```
4. Submit and debug.