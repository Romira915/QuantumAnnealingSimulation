package qa;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.http.HttpClient;
import java.util.ArrayList;
import java.util.List;

import org.apache.http.HttpStatus;
import org.apache.http.NameValuePair;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.message.BasicNameValuePair;

public class Webhook {
    private String url;
    private String username;
    private List<NameValuePair> json;

    public Webhook(String url, String name) {
        this.url = url;
        this.username = name;

        this.json.add(new BasicNameValuePair("username", this.username));
    }

    public Webhook(String url) {
        this(url, "SQA notification");
    }

    public boolean send(String content) {
        CloseableHttpClient httpClient = HttpClients.createDefault();
        HttpPost requestPost = new HttpPost(this.url);
        CloseableHttpResponse response = null;

        this.json.add(new BasicNameValuePair("content", content));
        try {
            requestPost.setEntity(new UrlEncodedFormEntity(this.json));
            response = httpClient.execute(requestPost);

            if (response.getStatusLine().getStatusCode() != HttpStatus.SC_OK) {
                return false;
            }
        } catch (UnsupportedEncodingException e) {
            System.err.println("Unsupport jsonCode");
            e.printStackTrace();
            // TODO: handle exception
        } catch (ClientProtocolException e) {
            System.err.println("Client Protocol Error");
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("IO Error");
            e.printStackTrace();
        } finally {
            try {
                if (httpClient != null) {
                    httpClient.close();
                }
                if (response != null) {
                    response.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
                // TODO: handle exception
            }
        }
        return true;
    }
}
