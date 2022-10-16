package com.example.facedetect;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.security.KeyManagementException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.Map;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLException;
import javax.net.ssl.SSLSession;
import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import javax.net.ssl.X509TrustManager;

public class SendPostAsyncTask extends AsyncTask<SendPostTaskParams, String, String> {
    private Context mContext;

    public SendPostAsyncTask(Context context) {
        mContext = context;
    }
    @Override
    protected String doInBackground(SendPostTaskParams... params) {
        String urlString = params[0].url;
        Map<String, Object> postDataParams = params[0].postData;
        TrustManagerFactory trustManager;
        BufferedInputStream inputStream = null;
        ByteArrayOutputStream responseArray = null;
        String response = "";
        try {
            URL url = new URL(urlString);
            // プライベート認証局のルート証明書でサーバー証明書を検証する
            // assets に格納しておいたプライベート証明書だけを含む KeyStore を設定
            KeyStore ks = KeyStoreUtil.getEmptyKeyStore();
            KeyStoreUtil.loadX509Certificate(ks,
                    mContext.getResources().getAssets().open("cacert.pem"));
            // ホスト名の検証を行う
            HttpsURLConnection.setDefaultHostnameVerifier(new HostnameVerifier() {
                @Override
                public boolean verify(String hostname, SSLSession session) {
                    if (!hostname.equals(session.getPeerHost())) {
                        return false;
                    }
                    return true;
                }
            });


            /// POST送信するデータをバイト配列で用意
            StringBuilder postData = new StringBuilder();
            for (Map.Entry<String, Object> entry : postDataParams.entrySet()) {
                if (postData.length() != 0) postData.append('&');
                postData.append(URLEncoder.encode(entry.getKey(), "UTF-8"))
                        .append('=')
                        .append(URLEncoder.encode(String.valueOf(entry.getValue()), "UTF-8"));
            }
            byte[] postDataBytes = postData.toString().getBytes("UTF-8");

            SSLContext sslCon = SSLContext.getInstance("TLS");

            trustManager = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
            trustManager.init(ks);
            sslCon.init(null, trustManager.getTrustManagers(), new SecureRandom());

            HttpURLConnection con = (HttpURLConnection) url.openConnection();
            con.setRequestMethod("POST");
            con.setDoOutput(true);
            con.setConnectTimeout(30000);
            HttpsURLConnection conn = (HttpsURLConnection)con;

            conn.setDefaultSSLSocketFactory(sslCon.getSocketFactory());
            conn.setSSLSocketFactory(sslCon.getSocketFactory());
            conn.getOutputStream().write(postDataBytes);

            int responseCode = conn.getResponseCode();

            if (responseCode == HttpsURLConnection.HTTP_OK) {
                String line;
                BufferedReader br=new BufferedReader(new InputStreamReader(conn.getInputStream()));
                while ((line=br.readLine()) != null) {
                    response += line;
                }
            }
        } catch(SSLException e) {
            // SSLException に対しユーザーに通知する等の適切な例外処理をする
            // サンプルにつき例外処理は割愛
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (Exception e) {
                    // 例外処理は割愛
                }
            }
            if (responseArray != null) {
                try {
                    responseArray.close();
                } catch (Exception e) {
                    // 例外処理は割愛

                }
            }
        }
        return "";
    }
}