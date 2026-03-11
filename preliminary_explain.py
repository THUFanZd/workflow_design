import random
import re
import time
import torch
from typing import Dict, List, Tuple, Optional
from model_with_sae import ModelWithSAEModule
import functools


class PreliminaryExplainer:
    def __init__(self, model_with_sae : ModelWithSAEModule, device: torch.device = "cuda"):
        self.model = model_with_sae.model
        self.sae = model_with_sae.sae
        self.hook_name = self.sae["__sae_lens_obj__"].cfg.metadata["hook_name"]
        self.act_hook_name = self.hook_name + ".hook_sae_acts_post"
        self.tokenizer = model_with_sae.tokenizer
        self.device = device
        self.feature_index = getattr(model_with_sae, "feature_index", None)
        self.layer = getattr(model_with_sae, "layer", None)
        self.use_hooked_transformer = getattr(model_with_sae, "use_hooked_transformer", False)
        self.VOCAB_PROJ_SYS_PROMPT = "You will be given a list of tokens related to a specific vector. These tokens represent a combination of embeddings that reconstruct the vector. Your task is to infer the most likely meaning or function of the vector based on these tokens. The list may include noise, such as unrelated terms, symbols, or programming jargon. Ignore whether the words are in multiple different languages, and do not mention it in your response. Focus on identifying a cohesive theme or concept shared by the most relevant tokens. Provide a specific sentence summarizing the meaning or function of the vector. Answer only with the summary. Avoid generic or overly broad answers, and disregard any noise in the list.\nVector 1\n    Tokens: ['contentLoaded', '▁hObject', ':✨', '▁AssemblyCulture', 'ContentAsync', '▁ivelany', '▁nahilalakip', 'IUrlHelper', '▁تضيفلها', '▁ErrIntOverflow'] ['▁could', 'could', '▁Could', 'Could', '▁COULD', '▁podría', '▁könnte', '▁podrían', '▁poderia', '▁könnten']\nExplanation of vector 1 behavior: this vector is related to the word could.\nVector 2\n    Tokens: ['▁CreateTagHelper', '▁ldc', 'PropertyChanging', '▁jsPsych', 'ulement', '▁IBOutlet', '▁wireType', '▁initComponents', '▁متعلقه', 'Бахар'] ['▁مشين', '▁charity', '▁donation', '▁charitable', '▁volont', '▁donations', 'iNdEx', 'Parcelize', 'DatabaseError', 'BufferException']\nExplanation of vector 2 behavior: this vector is related to charity and donations.\nVector 3\n    Tokens: ['▁tomorrow', '▁tonight', '▁yesterday', '▁today', 'yesterday', 'tomorrow', '▁demain', '▁Tomorrow', 'Tomorrow', '▁Yesterday'] ['▁Wex', 'ကိုးကား', 'Ārējās', 'piecze', ')$/,', '▁außer', '[]=$', 'cendental', 'ɜ', 'aderie']\nExplanation of vector 3 behavior: this vector is related to specific dates, like tomorrow, tonight and yesterday.\n\n"
        self.VOACB_PROJ_USER_PROMPT = "Vector 4\n    Tokens: {0}\nExplanation of vector 4 behavior: this vector is related to"
    
    @torch.no_grad()
    def vocab_proj(
        self,
        feature_index: Optional[int] = None,
        top_k: int = 10,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Project a single SAE decoder feature through final norm and unembedding.
        Returns topk, bottomk, and abs_topk lists of (token, logit, token_id).
        """
        if self.model is None or self.tokenizer is None or not self.sae:
            raise RuntimeError("Model, tokenizer, and SAE must be loaded.")

        if feature_index is None:
            raise RuntimeError("feature_index is required when not set on the object.")

        if top_k <= 0:
            return {"topk": [], "bottomk": [], "abs_topk": []}

        decoder_matrix: Optional[torch.Tensor] = None
        if "__sae_lens_obj__" in self.sae:
            sae_obj = self.sae["__sae_lens_obj__"]
            if hasattr(sae_obj, "W_dec"):
                decoder_matrix = sae_obj.W_dec
            elif hasattr(sae_obj, "decoder") and hasattr(sae_obj.decoder, "weight"):
                decoder_matrix = sae_obj.decoder.weight
        if decoder_matrix is None:
            if "W_dec" in self.sae:
                decoder_matrix = self.sae["W_dec"]
            elif "decoder.weight" in self.sae:
                decoder_matrix = self.sae["decoder.weight"]
            else:
                raise RuntimeError("SAE decoder matrix not available.")
            

        unembed_weight: Optional[torch.Tensor] = None
        unembed_bias: Optional[torch.Tensor] = None

        if hasattr(self.model, "unembed") and hasattr(self.model.unembed, "W_U"):
            unembed_weight = self.model.unembed.W_U
            if hasattr(self.model.unembed, "b_U"):
                unembed_bias = self.model.unembed.b_U
        if unembed_weight is None and hasattr(self.model, "W_U"):
            unembed_weight = self.model.W_U
        if unembed_weight is None and hasattr(self.model, "get_output_embeddings"):
            out_embed = self.model.get_output_embeddings()
            if out_embed is not None and hasattr(out_embed, "weight"):
                unembed_weight = out_embed.weight
                if hasattr(out_embed, "bias") and out_embed.bias is not None:
                    unembed_bias = out_embed.bias
        if unembed_weight is None and hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
            unembed_weight = self.model.lm_head.weight
            if hasattr(self.model.lm_head, "bias") and self.model.lm_head.bias is not None:
                unembed_bias = self.model.lm_head.bias

        if unembed_weight is None:
            raise RuntimeError("Unembedding matrix not available on the model.")

        unembed_weight = unembed_weight.to(self.device)
        if unembed_weight.ndim != 2:
            raise RuntimeError("Unembedding matrix must be 2D.")

        tokenizer_vocab = getattr(self.tokenizer, "vocab_size", None)
        if tokenizer_vocab is not None and unembed_weight.shape[0] == tokenizer_vocab:
            unembed_matrix = unembed_weight.T
            vocab_size = unembed_weight.shape[0]
            d_model = unembed_weight.shape[1]
        elif tokenizer_vocab is not None and unembed_weight.shape[1] == tokenizer_vocab:
            unembed_matrix = unembed_weight
            vocab_size = unembed_weight.shape[1]
            d_model = unembed_weight.shape[0]
        elif unembed_weight.shape[0] < unembed_weight.shape[1]:
            unembed_matrix = unembed_weight
            vocab_size = unembed_weight.shape[1]
            d_model = unembed_weight.shape[0]
        else:
            unembed_matrix = unembed_weight.T
            vocab_size = unembed_weight.shape[0]
            d_model = unembed_weight.shape[1]

        decoder_matrix = decoder_matrix.to(self.device)
        if decoder_matrix.ndim != 2:
            raise RuntimeError("SAE decoder matrix must be 2D.")

        feature_vec = decoder_matrix[feature_index]

        feature_vec = feature_vec.to(device=self.device, dtype=unembed_matrix.dtype)

        ln_final = None
        for attr_path in (
            "ln_final",
            "ln_f",
            "final_layernorm",
            "norm",
            "model.norm",
            "model.ln_f",
            "model.ln_final",
            "transformer.ln_f",
            "transformer.ln_final",
        ):
            current = self.model
            for part in attr_path.split("."):
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    current = None
                    break
            if current is not None:
                ln_final = current
                break

        if ln_final is not None:
            feature_vec = ln_final(feature_vec.unsqueeze(0)).squeeze(0)

        logits = torch.matmul(feature_vec, unembed_matrix)
        if unembed_bias is not None and unembed_bias.numel() == vocab_size:
            logits = logits + unembed_bias.to(device=logits.device, dtype=logits.dtype)

        logits_cpu = logits.detach().float().cpu()
        top_k = min(top_k, vocab_size)

        if skip_special_tokens and hasattr(self.tokenizer, "all_special_ids"):
            top_logits = logits_cpu.clone()
            bottom_logits = logits_cpu.clone()
            abs_logits = logits_cpu.abs()
            for special_id in self.tokenizer.all_special_ids:
                if 0 <= special_id < vocab_size:
                    top_logits[special_id] = float("-inf")
                    bottom_logits[special_id] = float("inf")
                    abs_logits[special_id] = float("-inf")
        else:
            top_logits = logits_cpu
            bottom_logits = logits_cpu
            abs_logits = logits_cpu.abs()

        top_vals, top_ids = torch.topk(top_logits, k=top_k)
        bottom_vals, bottom_ids = torch.topk(bottom_logits, k=top_k, largest=False)
        abs_vals, abs_ids = torch.topk(abs_logits, k=top_k)

        top_ids_list = top_ids.tolist()
        bottom_ids_list = bottom_ids.tolist()
        abs_ids_list = abs_ids.tolist()

        top_tokens = self.tokenizer.convert_ids_to_tokens(top_ids_list)
        bottom_tokens = self.tokenizer.convert_ids_to_tokens(bottom_ids_list)
        abs_tokens = self.tokenizer.convert_ids_to_tokens(abs_ids_list)

        abs_actual_vals = logits_cpu[abs_ids].tolist()

        return {
            "topk": list(zip(top_tokens, top_vals.tolist(), top_ids_list)),
            "bottomk": list(zip(bottom_tokens, bottom_vals.tolist(), bottom_ids_list)),
            "abs_topk": list(zip(abs_tokens, abs_actual_vals, abs_ids_list)),
        }
    
    @torch.no_grad()
    def token_change(
        self,
        feature_index: Optional[int] = None,
        corpus: Optional[List[str]] = None,
        top_k: int = 10,
        batch_size: int = 32,
        value: Optional[float] = 10.0,
        skip_special_tokens: bool = True,
    ) -> Dict[str, List[Tuple[str, float, int]]]:
        """
        Estimate token logit changes when clamping a feature to +value.
        Returns combined tokens with increased/decreased logits for steering hints.
        """
        token_data = self.token_change_split(
            feature_index=feature_index,
            corpus=corpus,
            top_k=top_k,
            batch_size=batch_size,
            value=value,
            skip_special_tokens=skip_special_tokens,
        )

        return token_data["pos_toks"] + token_data["neg_toks"]

    @torch.no_grad()
    def token_change_split(
        self,
        feature_index: Optional[int] = None,
        corpus: Optional[List[str]] = None,
        top_k: int = 10,
        batch_size: int = 32,
        value: Optional[float] = 10.0,
        skip_special_tokens: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Estimate token logit changes when clamping a feature to +value.
        Returns separate pos/neg token lists for steering analysis.
        """
        if self.model is None or self.tokenizer is None or not self.sae:
            raise RuntimeError("Model, tokenizer, and SAE must be loaded.")
        if corpus is None or len(corpus) == 0:
            raise RuntimeError("corpus must be a non-empty list of strings.")
        if feature_index is None:
            raise RuntimeError("feature_index is required when not set on the object.")

        if top_k <= 0:
            return {"pos_toks": [], "neg_toks": []}

        if batch_size <= 0:
            raise RuntimeError("batch_size must be a positive integer.")

        def set_feature_act_hook(act, hook, feature, value):
            act[:, :, feature] = value

        sae_obj = self.sae["__sae_lens_obj__"]

        pos_sum_logits = None
        neg_sum_logits = None
        total_tokens = 0
        corpus_is_tensor = isinstance(corpus, torch.Tensor)

        for start in range(0, len(corpus), batch_size):
            batch = corpus[start:start + batch_size]
            if corpus_is_tensor:
                input_ids = batch.to(self.device)
            else:
                input_ids = torch.tensor(batch, dtype=torch.long, device=self.device)

            clean_logits = self.model.run_with_saes(input_ids, saes=[sae_obj])

            pos_inter_logits = self.model.run_with_hooks_with_saes(
                input_ids,
                saes=[sae_obj],
                fwd_hooks=[(self.act_hook_name, functools.partial(set_feature_act_hook, feature=feature_index, value=value))],
            )

            neg_inter_logits = self.model.run_with_hooks_with_saes(
                input_ids,
                saes=[sae_obj],
                fwd_hooks=[(self.act_hook_name, functools.partial(set_feature_act_hook, feature=feature_index, value=-value))],
            )

            pos_diff = pos_inter_logits - clean_logits
            neg_diff = neg_inter_logits - clean_logits

            batch_tokens = pos_diff.shape[0] * pos_diff.shape[1]
            total_tokens += batch_tokens

            pos_batch_sum = pos_diff.sum(dim=(0, 1))
            neg_batch_sum = neg_diff.sum(dim=(0, 1))

            if pos_sum_logits is None:
                pos_sum_logits = pos_batch_sum
            else:
                pos_sum_logits = pos_sum_logits + pos_batch_sum
            if neg_sum_logits is None:
                neg_sum_logits = neg_batch_sum
            else:
                neg_sum_logits = neg_sum_logits + neg_batch_sum

        if total_tokens == 0:
            raise RuntimeError("No tokens were processed in token_change_split.")

        pos_diff_logits = pos_sum_logits / total_tokens
        neg_diff_logits = neg_sum_logits / total_tokens

        def to_tokens(ids):
            if hasattr(self.model, "to_str_tokens"):
                return self.model.to_str_tokens(ids)
            ids_list = ids.tolist() if torch.is_tensor(ids) else list(ids)
            return self.tokenizer.convert_ids_to_tokens(ids_list)

        pos_high_ids = pos_diff_logits.topk(top_k).indices
        pos_low_ids = pos_diff_logits.topk(top_k, largest=False).indices
        neg_high_ids = neg_diff_logits.topk(top_k).indices
        neg_low_ids = neg_diff_logits.topk(top_k, largest=False).indices

        neg_toks = to_tokens(pos_high_ids) + to_tokens(neg_low_ids)
        pos_toks = to_tokens(pos_low_ids) + to_tokens(neg_high_ids)

        return {"pos_toks": pos_toks, "neg_toks": neg_toks}

    def _get_embedding_matrix(self) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        if hasattr(self.model, "get_input_embeddings"):
            embed = self.model.get_input_embeddings()
            if embed is not None and hasattr(embed, "weight"):
                return embed.weight

        if hasattr(self.model, "W_E"):
            return self.model.W_E  # type: ignore[attr-defined]

        if hasattr(self.model, "embed") and hasattr(self.model.embed, "W_E"):
            return self.model.embed.W_E  # type: ignore[attr-defined]

        raise RuntimeError("Could not locate embedding matrix on the model.")

    @torch.no_grad()
    def _get_feature_activations(
        self,
        input_ids: torch.Tensor,
        feature_index: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not self.sae:
            raise RuntimeError("No SAE loaded.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq].")
        if feature_index < 0:
            raise ValueError("feature_index must be non-negative.")
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask must be 2D with shape [batch, seq].")
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must match input_ids shape.")

        def _get_layer_activations() -> torch.Tensor:
            if self.use_hooked_transformer:
                if self.hook_name is None:
                    raise RuntimeError("hook_name is required for HookedTransformer path.")
                _, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=[self.hook_name],
                )
                return cache[self.hook_name]

            if self.layer is None:
                raise RuntimeError("layer is required for HuggingFace forward path.")
            hf_attention_mask = (
                attention_mask.to(device=input_ids.device, dtype=torch.long)
                if attention_mask is not None
                else torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=hf_attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None or self.layer >= len(hidden_states):
                raise RuntimeError(f"Layer {self.layer} not available.")
            return hidden_states[self.layer]

        layer_activations = _get_layer_activations()

        if "__sae_lens_obj__" in self.sae:
            sae_obj = self.sae["__sae_lens_obj__"]
            sae_features = sae_obj.encode(layer_activations)
            if sae_features is None or sae_features.ndim != 3:
                raise RuntimeError("SAE encode output must be 3D [batch, seq, n_features].")
            if feature_index >= sae_features.shape[-1]:
                raise ValueError(f"Feature {feature_index} out of range.")
            return sae_features[:, :, feature_index].detach().to(torch.float32)

        if "encoder.weight" in self.sae:
            encoder_weight = self.sae["encoder.weight"].to(self.device)
            if feature_index >= encoder_weight.shape[0]:
                raise ValueError(f"Feature {feature_index} out of range.")
            flat = layer_activations.reshape(-1, layer_activations.shape[-1])
            encoder_vec = encoder_weight[feature_index].to(device=flat.device, dtype=flat.dtype)
            feature_vals = torch.matmul(flat, encoder_vec)
            return feature_vals.view(input_ids.shape[0], input_ids.shape[1]).detach().to(torch.float32)

        raise RuntimeError("SAE not properly loaded.")

    @torch.no_grad()
    def count_bos_vocab_activation_fragments(
        self,
        feature_index: Optional[int] = None,
        activation_threshold: float = 0.0,
        batch_size: int = 256,
        top_k: int = 20,
        include_special_tokens: bool = True,
        return_all_token_stats: bool = False,
    ) -> Dict[str, object]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        if self.model is None or not self.sae:
            raise RuntimeError("Model and SAE must be loaded.")

        feature_index = self.feature_index if feature_index is None else int(feature_index)
        if feature_index is None:
            raise RuntimeError("feature_index is required when not set on the object.")
        feature_index = int(feature_index)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not hasattr(self.tokenizer, "vocab_size") or self.tokenizer.vocab_size is None:
            raise RuntimeError("Tokenizer vocab_size is required.")
        vocab_size = int(self.tokenizer.vocab_size)
        if vocab_size <= 0:
            raise RuntimeError("Tokenizer vocab_size must be positive.")

        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is None:
            raise RuntimeError("Tokenizer bos_token_id is required.")
        bos_token_id = int(bos_token_id)

        token_ids = list(range(vocab_size))
        if not include_special_tokens and hasattr(self.tokenizer, "all_special_ids"):
            special_ids = {
                int(tok_id)
                for tok_id in self.tokenizer.all_special_ids
                if 0 <= int(tok_id) < vocab_size
            }
            token_ids = [tok_id for tok_id in token_ids if tok_id not in special_ids]

        evaluated_token_count = len(token_ids)
        if evaluated_token_count == 0:
            result: Dict[str, object] = {
                "evaluated_token_count": 0,
                "vocab_size": vocab_size,
                "bos_token_id": bos_token_id,
                "activation_threshold": float(activation_threshold),
                "total_activation_fragments": 0,
                "activated_sequence_count": 0,
                "activation_rate": 0.0,
                "topk_sequences_by_max_activation": [],
            }
            if return_all_token_stats:
                result["all_token_stats"] = {
                    "token_ids": [],
                    "max_activations": [],
                    "fragment_counts": [],
                }
            return result

        total_activation_fragments = 0
        activated_sequence_count = 0
        max_activations: List[float] = []
        frag_counts_all: List[int] = []
        evaluated_token_ids: List[int] = []

        for start in range(0, evaluated_token_count, batch_size):
            batch_token_ids = token_ids[start:start + batch_size]
            batch_len = len(batch_token_ids)

            bos_column = torch.full(
                (batch_len, 1),
                fill_value=bos_token_id,
                dtype=torch.long,
                device=self.device,
            )
            token_column = torch.tensor(
                batch_token_ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(1)
            input_ids = torch.cat((bos_column, token_column), dim=1)

            feature_acts = self._get_feature_activations(
                input_ids=input_ids,
                feature_index=feature_index,
            )
            if feature_acts.ndim != 2:
                raise RuntimeError("Feature activations must be 2D [batch, seq].")
            if feature_acts.shape[0] != input_ids.shape[0] or feature_acts.shape[1] != input_ids.shape[1]:
                raise RuntimeError(
                    "Feature activation shape must match input_ids shape."
                )

            active = feature_acts > activation_threshold
            starts = active[:, :1] | (active[:, 1:] & ~active[:, :-1])
            frag_counts = starts.sum(dim=1)
            max_act = feature_acts.max(dim=1).values

            total_activation_fragments += int(frag_counts.sum().item())
            activated_sequence_count += int((frag_counts > 0).sum().item())

            max_activations.extend(max_act.detach().cpu().tolist())
            frag_counts_all.extend([int(x) for x in frag_counts.detach().cpu().tolist()])
            evaluated_token_ids.extend(batch_token_ids)

        activation_rate = activated_sequence_count / evaluated_token_count
        top_k_eff = min(top_k, evaluated_token_count)

        max_activations_tensor = torch.tensor(max_activations, dtype=torch.float32)
        top_vals, top_indices = torch.topk(max_activations_tensor, k=top_k_eff)
        top_indices_list = top_indices.tolist()
        top_token_ids = [evaluated_token_ids[idx] for idx in top_indices_list]
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_token_ids)
        top_frag_counts = [frag_counts_all[idx] for idx in top_indices_list]

        topk_sequences = list(
            zip(
                top_tokens,
                top_vals.tolist(),
                top_token_ids,
                top_frag_counts,
            )
        )

        result = {
            "evaluated_token_count": evaluated_token_count,
            "vocab_size": vocab_size,
            "bos_token_id": bos_token_id,
            "activation_threshold": float(activation_threshold),
            "total_activation_fragments": int(total_activation_fragments),
            "activated_sequence_count": int(activated_sequence_count),
            "activation_rate": float(activation_rate),
            "topk_sequences_by_max_activation": topk_sequences,
        }
        if return_all_token_stats:
            result["all_token_stats"] = {
                "token_ids": [int(x) for x in evaluated_token_ids],
                "max_activations": [float(x) for x in max_activations],
                "fragment_counts": [int(x) for x in frag_counts_all],
            }
        return result

    @torch.no_grad()
    def analyze_sentence_activation_fragments(
        self,
        sentences: List[str],  # Note 全量语料
        feature_index: Optional[int] = None,
        activation_threshold: float = 0.0,
        batch_size: int = 32,
        max_length: int = 256,
        top_tokens_per_sentence: int = 5,
        include_special_tokens: bool = False,
    ) -> List[Dict[str, object]]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        if self.model is None or not self.sae:
            raise RuntimeError("Model and SAE must be loaded.")

        if not isinstance(sentences, list) or len(sentences) == 0:
            raise ValueError("sentences must be a non-empty list of strings.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if top_tokens_per_sentence <= 0:
            raise ValueError("top_tokens_per_sentence must be positive.")

        feature_index = self.feature_index if feature_index is None else int(feature_index)
        if feature_index is None:
            raise RuntimeError("feature_index is required when not set on the object.")
        feature_index = int(feature_index)

        normalized_sentences: List[str] = []
        for sentence in sentences:
            if not isinstance(sentence, str):
                raise ValueError("All sentences must be strings.")
            if sentence.strip() == "":
                raise ValueError("Empty sentence is not allowed.")
            normalized_sentences.append(sentence)

        special_ids = set()
        if hasattr(self.tokenizer, "all_special_ids") and self.tokenizer.all_special_ids is not None:
            special_ids = {int(tok_id) for tok_id in self.tokenizer.all_special_ids}

        results: List[Dict[str, object]] = []
        for start in range(0, len(normalized_sentences), batch_size):
            batch_sentences = normalized_sentences[start:start + batch_size]
            encoded = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)

            feature_acts = self._get_feature_activations(
                input_ids=input_ids,
                feature_index=feature_index,
                attention_mask=attention_mask,
            )
            if feature_acts.shape != input_ids.shape:
                raise RuntimeError("Feature activation shape must match input_ids shape.")

            input_ids_cpu = input_ids.detach().cpu()
            attention_mask_cpu = attention_mask.detach().cpu().bool()
            feature_acts_cpu = feature_acts.detach().cpu()

            for row_idx, sentence in enumerate(batch_sentences):
                valid_mask = attention_mask_cpu[row_idx]
                valid_token_ids = input_ids_cpu[row_idx][valid_mask].tolist()
                valid_tokens = [str(tok) for tok in self.tokenizer.convert_ids_to_tokens(valid_token_ids)]
                valid_acts = [float(x) for x in feature_acts_cpu[row_idx][valid_mask].tolist()]

                if valid_acts:
                    max_activation = float(max(valid_acts))
                    mean_activation = float(sum(valid_acts) / len(valid_acts))
                    sum_activation = float(sum(valid_acts))
                else:
                    max_activation = 0.0
                    mean_activation = 0.0
                    sum_activation = 0.0

                fragments: List[Dict[str, object]] = []
                fragment_start = None
                for pos, act_val in enumerate(valid_acts):
                    is_active = act_val > activation_threshold  # Note 阈值何来
                    if is_active and fragment_start is None:
                        fragment_start = pos
                    if (not is_active) and fragment_start is not None:
                        fragment_end = pos - 1
                        frag_acts = valid_acts[fragment_start:fragment_end + 1]
                        fragments.append(
                            {
                                "start": int(fragment_start),
                                "end": int(fragment_end),
                                "tokens": valid_tokens[fragment_start:fragment_end + 1],
                                "activations": [float(x) for x in frag_acts],
                                "max_activation": float(max(frag_acts)) if frag_acts else 0.0,
                            }
                        )
                        fragment_start = None
                if fragment_start is not None:
                    fragment_end = len(valid_acts) - 1
                    frag_acts = valid_acts[fragment_start:fragment_end + 1]
                    fragments.append(
                        {
                            "start": int(fragment_start),
                            "end": int(fragment_end),
                            "tokens": valid_tokens[fragment_start:fragment_end + 1],
                            "activations": [float(x) for x in frag_acts],
                            "max_activation": float(max(frag_acts)) if frag_acts else 0.0,
                        }
                    )

                candidate_indices = [
                    idx
                    for idx in range(len(valid_acts))
                    if include_special_tokens or valid_token_ids[idx] not in special_ids
                ]
                candidate_indices.sort(key=lambda idx: valid_acts[idx], reverse=True)
                top_indices = candidate_indices[:top_tokens_per_sentence]
                top_tokens = [
                    {
                        "token": valid_tokens[idx],
                        "index": int(idx),
                        "activation": float(valid_acts[idx]),
                    }
                    for idx in top_indices
                ]

                results.append(
                    {
                        "sentence": sentence,
                        "max_activation": float(max_activation),
                        "mean_activation": float(mean_activation),
                        "sum_activation": float(sum_activation),
                        "tokens": valid_tokens,
                        "per_token_activations": [float(x) for x in valid_acts],
                        "fragment_count": int(len(fragments)),
                        "fragments": fragments,
                        "top_tokens": top_tokens,
                    }
                )

        return results

    @torch.no_grad()
    def embedding_lens(
        self,
        feature_index: Optional[int] = None,
        top_k: int = 10,
        batch_size: int = 256,
        skip_special_tokens: bool = True,
    ) -> List[Tuple[str, float, int]]:
        """
        Encode each token embedding with the SAE encoder and return top-k tokens
        for the selected feature. Returns (token, activation, token_id).
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        if not self.sae:
            raise RuntimeError("No SAE loaded.")

        feature_index = self.feature_index if feature_index is None else int(feature_index)
        if top_k <= 0:
            return []

        embedding_weight = self._get_embedding_matrix().detach()
        vocab_size = embedding_weight.shape[0]
        top_k = min(top_k, vocab_size)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        activations = torch.empty(vocab_size, device="cpu", dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, vocab_size, batch_size):
                end = min(start + batch_size, vocab_size)
                chunk = embedding_weight[start:end].to(self.device)

                if self.sae and "__sae_lens_obj__" in self.sae:
                    sae_obj = self.sae["__sae_lens_obj__"]
                    if chunk.ndim == 2:
                        chunk = chunk.unsqueeze(0)
                    sae_features = sae_obj.encode(chunk)  # [1, chunk, n_features]
                    if feature_index >= sae_features.shape[-1]:
                        raise ValueError(f"Feature {feature_index} out of range")
                    chunk_act = sae_features[0, :, feature_index]
                elif self.sae and "encoder.weight" in self.sae:
                    encoder_weight = self.sae["encoder.weight"].to(self.device)
                    if feature_index >= encoder_weight.shape[0]:
                        raise ValueError(f"Feature {feature_index} out of range")
                    chunk_act = torch.matmul(chunk, encoder_weight[feature_index])
                else:
                    raise RuntimeError("SAE not properly loaded")

                activations[start:end] = chunk_act.detach().float().cpu()

        if skip_special_tokens and hasattr(self.tokenizer, "all_special_ids"):
            for special_id in self.tokenizer.all_special_ids:
                if 0 <= special_id < vocab_size:
                    activations[special_id] = float("-inf")

        top_vals, top_ids = torch.topk(activations, k=top_k)
        top_ids_list = top_ids.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(top_ids_list)
        
        return list(zip(tokens, top_vals.tolist(), top_ids_list))

    def collect_word_list(
        self,
        feature_index: Optional[int] = None,
        corpus: Optional[List[List[int]]] = None,
        top_k: int = 10,
        token_change_value: float = 5.0,
        include_token_change: bool = True,
        skip_special_tokens: bool = True,
        normalize_for_prompt: bool = True,
    ) -> List[str]:
        tokens: List[str] = []

        vocab_proj_data = self.vocab_proj(
            feature_index=feature_index,
            top_k=top_k,
            skip_special_tokens=skip_special_tokens,
        )
        tokens.extend([t for t, _, _ in vocab_proj_data.get("topk", [])])
        tokens.extend([t for t, _, _ in vocab_proj_data.get("bottomk", [])])

        if include_token_change and corpus:
            token_change_tokens = self.token_change(
                    feature_index=feature_index,
                    corpus=corpus,
                    top_k=top_k,
                    value=token_change_value,
                    skip_special_tokens=skip_special_tokens,
                )
            tokens.extend(token_change_tokens)

        embedding_tokens = self.embedding_lens(
            feature_index=feature_index,
            top_k=top_k,
            skip_special_tokens=skip_special_tokens,
        )
        tokens.extend([t for t, _, _ in embedding_tokens])

        return self._dedupe_tokens(tokens, normalize_for_prompt=normalize_for_prompt)

    def generate_sequences_from_word_list(
        self,
        word_list: List[str],
        num_sequences: int,
        llm_model: str,
        max_tokens_per_sequence: Optional[int] = None,
        max_retries: int = 3,
        retry_delay_s: float = 4.0,
        generator_llm: str = "glm-4.5",
    ) -> List[str]:
        if not word_list:
            raise RuntimeError("word_list must be non-empty to generate sequences.")
        if num_sequences <= 0:
            raise RuntimeError("num_sequences must be positive.")
        if num_sequences > len(word_list):
            print(
                f"Warning: num_sequences ({num_sequences}) exceeds word_list size "
                f"({len(word_list)}); extra sentences will sample tokens from the word list."
            )

        token_groups = self._distribute_tokens(word_list, num_sequences, max_tokens_per_sequence)
        prompt = self._build_sequence_prompt(token_groups, num_sequences)

        from agents.agent import agent as global_agent, validate_agent_response

        agent = global_agent
        history = [
            {
                "role": "system",
                "content": (
                    "You must output exactly "
                    + str(num_sequences)
                    + " numbered sentences (1.."
                    + str(num_sequences)
                    + "). Each sentence must include the required tokens verbatim. "
                    "Output only the numbered sentences with no extra commentary."
                    + "YOU MUST CONFIRM THAT EACH SENTENCE MUST BE AT LEAST 15 WORDS LONG."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        attempts = max(1, int(max_retries) + 1)
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = agent.ask_agent(generator_llm, history, max_tokens=8192)
            except Exception as exc:
                last_error = exc
                response = ""

            print(response)
            sequences = self._parse_sequence_response(response, num_sequences)
            if len(sequences) >= num_sequences:
                return sequences[:num_sequences]
            if sequences:
                print(
                    "LLM sequence generation returned "
                    f"{len(sequences)}/{num_sequences} sequences; retrying."
                )

            if attempt < attempts:
                delay = retry_delay_s * (2 ** (attempt - 1))
                print(
                    "LLM sequence generation failed "
                    f"(attempt {attempt}/{attempts}); retrying in {delay:.1f}s."
                )
                time.sleep(delay)

        if last_error is not None:
            raise RuntimeError(
                f"LLM sequence generation failed after {attempts} attempts: {last_error}"
            )
        raise RuntimeError(
            "LLM sequence generation failed after "
            f"{attempts} attempts: insufficient sequences."
        )

    def _normalize_token_for_prompt(self, token: str) -> Optional[str]:
        if token is None:
            return None
        clean = token.strip()
        if not clean:
            return None
        if self.tokenizer is not None and hasattr(self.tokenizer, "all_special_tokens"):
            if clean in self.tokenizer.all_special_tokens:
                return None
        if clean.startswith("\u2581") or clean.startswith("\u0120"):
            clean = clean[1:]
        if clean.startswith("##"):
            clean = clean[2:]
        clean = clean.strip()
        if not clean:
            return None
        return clean

    def _dedupe_tokens(self, tokens: List[str], normalize_for_prompt: bool = True) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for token in tokens:
            cleaned = self._normalize_token_for_prompt(token) if normalize_for_prompt else token
            if not cleaned:
                continue
            if cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        return deduped

    def _distribute_tokens(
        self,
        word_list: List[str],
        num_sequences: int,
        max_tokens_per_sequence: Optional[int],
    ) -> List[List[str]]:
        groups: List[List[str]] = [[] for _ in range(num_sequences)]
        for idx, token in enumerate(word_list):
            groups[idx % num_sequences].append(token)

        if word_list:
            for group in groups:
                if not group:
                    if len(word_list) >= 3:
                        group.extend(random.sample(word_list, 3))
                    else:
                        group.extend(random.choices(word_list, k=3))

        if max_tokens_per_sequence and max_tokens_per_sequence > 0:
            groups = [group[:max_tokens_per_sequence] for group in groups]

        return groups

    def _build_sequence_prompt(
        self,
        token_groups: List[List[str]],
        num_sequences: int,
    ) -> str:
        
        lines = [f"{idx}) " + ", ".join(tokens) for idx, tokens in enumerate(token_groups, start=1)]
        tokens_block = "\n".join(lines)

        prompt = (
            "Generate exactly "
            + str(num_sequences)
            + " natural language sentences.\n"
            "Each sentence must include all explicit tokens listed for it (verbatim, case-sensitive).\n"
            "EACH SENTENCE MUST BE AT LEAST 15 WORDS LONG.\n"
            "Make the sentences DIVERSE in context, subject, and style.\n\n"
            "Example:\n"
            "Tokens to include by sentence:\n"
            "1) lantern, harbor\n"
            "2) recipe, limestone\n\n"
            "Example output:\n"
            "1. The lantern bobbed in the harbor as the fog rolled in, guiding the late ship home.\n"
            "2. She copied the recipe onto limestone tiles for safekeeping, then hid them behind a loose brick.\n\n"
            "Now your task.\n"
            "Tokens to include by sentence:\n"
            + tokens_block
            + "\n\n"
        )

        return (
            prompt
            + "OUTPUT FORMAT:\n"
            "1. [sentence]\n"
            "2. [sentence]\n"
            "... up to "
            + str(num_sequences)
            + "\n"
            "No extra commentary."
        )

    def _parse_sequence_response(self, response: str, num_sequences: int) -> List[str]:
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        sequences: List[str] = []
        for line in lines:
            match = re.match(r"^\s*\d+[\.\)]\s*(.+)$", line)
            if match:
                candidate = match.group(1).strip()
            else:
                candidate = line.strip()
            if candidate:
                sequences.append(candidate)
        if num_sequences > 0:
            return sequences[:num_sequences]
        return sequences
        
