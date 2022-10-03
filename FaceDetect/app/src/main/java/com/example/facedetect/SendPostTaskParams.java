package com.example.facedetect;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

class SendPostTaskParams {
    String url;
    Map<String, Object> postData;

    public SendPostTaskParams(String url, HashMap<String, Object> postData) {
        this.url = url;
        this.postData = postData;
    }

    public void MyTaskParams(String url, LinkedHashMap<String, Object> postData) {
        this.url = url;
        this.postData = postData;
    }
}