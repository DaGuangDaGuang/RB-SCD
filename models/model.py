    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:

        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        xA = list(xA[0:4])
        xB = list(xB[0:4])

        xA3L, xA3H = self.yHL(xA[3])
        xA3L = self.outconv_bn_relu_L3(xA3L)
        xA3L = F.interpolate(xA3L, size=(8, 8), mode='bilinear', align_corners=False)

        xA2L, xA2H = self.yHL(xA[2])
        xA2L = self.outconv_bn_relu_L2(xA2L)
        xA2L = F.interpolate(xA2L, size=(16, 16), mode='bilinear', align_corners=False)

        xB3L, xB3H = self.yHL(xB[3])
        xB3L = self.outconv_bn_relu_L3(xB3L)
        xB3L = F.interpolate(xB3L, size=(8, 8), mode='bilinear', align_corners=False)

        xB2L, xB2H = self.yHL(xB[2])
        xB2L = self.outconv_bn_relu_L2(xB2L)
        xB2L = F.interpolate(xB2L, size=(16, 16), mode='bilinear', align_corners=False)

        xA1L, xA1H = self.yHL(xA[1])
        xA1H = self.conv_bn_relu_1(xA1H)
        xA0L, xA0H = self.yHL(xA[0])
        xA0H = self.conv_bn_relu_0(xA0H)
        xB1L, xB1H = self.yHL(xB[1])
        xB1H = self.conv_bn_relu_1(xB1H)
        xB0L, xB0H = self.yHL(xB[0])
        xB0H = self.conv_bn_relu_0(xB0H)


        xA0H = F.interpolate(xA0H, size=(64, 64), mode='bilinear', align_corners=False)
        xA1H = F.interpolate(xA1H, size=(32, 32), mode='bilinear', align_corners=False)
        xB0H = F.interpolate(xB0H, size=(64, 64), mode='bilinear', align_corners=False)
        xB1H = F.interpolate(xB1H, size=(32, 32), mode='bilinear', align_corners=False)

        xA[3] = self.conv3(torch.cat([xA[3], xB3L], dim=1)) + xA[3]
        xA[2] = self.conv2(torch.cat([xA[2], xB2L], dim=1)) + xA[2]
        xA[1] = self.conv1(torch.cat([xA[1], xA1H], dim=1)) + xA[1]
        xA[0] = self.conv0(torch.cat([xA[0], xA0H], dim=1)) + xA[0]

        xB[3] = self.conv3(torch.cat([xB[3], xA3L], dim=1)) + xB[3]
        xB[2] = self.conv2(torch.cat([xB[2], xA2L], dim=1)) + xB[2]
        xB[1] = self.conv1(torch.cat([xB[1], xB1H], dim=1)) + xB[1]
        xB[0] = self.conv0(torch.cat([xB[0], xB0H], dim=1)) + xB[0]
        x_globalA, x_localA = self.attnpool(xA[3])
        xA.append([x_globalA, x_localA])
        x_globalB, x_localB = self.attnpool(xB[3])
        xB.append([x_globalB, x_localB])
        textA, textB = self.get_cls_text(batch_img_metas, False)
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]

        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i] - x_clipB[i])) for i in range(len(x_clipA))]
        x_diff = [F.sigmoid(1 - torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) for i in
                  range(len(x_clipA))]

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddingsA, ] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i] * x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, batch_img_metas,
                                                        self.test_cfg)

        return seg_logits
def after_extract_feat_clip(self, x, text):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                       dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text = text.to(global_feat.device)
        contexts_ = torch.cat([self.contexts2] * int(x[0].size()[0]), dim=0).to(global_feat.device)
        text_embeddings = self.text_encoder(text.to(global_feat.device), contexts_).expand(B, -1, -1).to(global_feat.device)
        text_embeddings_org = text_embeddings
        text_embeddings_freq = torch.fft.rfft(text_embeddings, dim=1, norm='ortho')

        edge_index = self.build_edge_index(text_embeddings_freq)
        x = self.prepare_node_features(text_embeddings_freq)

        filtered_features = []
        for i in range(3):
            graph_conv_layer = self.graph_conv_layers[i]
            filtered_feature = graph_conv_layer(x, edge_index)
            filtered_features.append(filtered_feature)
        C = torch.stack(filtered_features)  # (filter, batch, s, dim)
        filtered_features = torch.sum(C, dim=0)  # (batch, s, dim)
        filtered_features = self.linear_layer(filtered_features.view(-1, 2048)).view(filtered_features.size(0),
                                                                                filtered_features.size(1), 1024)
        text_diff = self.context_decoder(filtered_features, visual_context)
        text_embeddings = text_diff + self.linear_layer2(text_embeddings.view(-1, 6 * 1024)).view(text_embeddings.size(0),4,1024)
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)

        return  text_embeddings, x_orig, score_map
def yHL(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        return yL, yH
      def get_cls_text(self, img_infos, train=True):
        textA = []
        textB = []
        for i in range(len(img_infos)):
            probabilitiesA = img_infos[i].get('jsonA', {})
            probabilitiesB = img_infos[i].get('jsonB', {})

            foreA = self.build_foreground_text(probabilitiesA, 'T1')

            foreB = self.build_foreground_text(probabilitiesB, 'T2')
            backA = ', '.join(['remote sensing image background objects'])
            backB = ', '.join(['remote sensing image background objects'])

            textA.append(
                torch.cat([tokenize(c, context_length=self.context_length) for c in [backA, foreA[0],foreA[1],foreA[2],foreA[1],foreA[2]]]).unsqueeze(0))
            textB.append(
                torch.cat([tokenize(c, context_length=self.context_length) for c in [backB, foreB[0],foreB[1],foreB[2],foreB[1],foreB[2]]]).unsqueeze(0))


        return torch.cat(textA, dim=0), torch.cat(textB, dim=0)

def build_foreground_text(self, probabilities, time_period):
        result = []
        for key, value in probabilities.items():
            if key in ['building', 'parking lot', 'road', 'bridge', 'farmland', 'water', 'grass', 'vegtation',
                       'bare land']:
                result.append(
                    f'remote sensing image at time {time_period} have a {value} probability of being the {key}')
        return result
